from contextvars import ContextVar
from datetime import datetime
import json
from typing import Dict, Any, Optional, List
import uuid
from dataclasses import dataclass, asdict
import logging
from functools import wraps

@dataclass
class Span:
    span_id: str
    parent_span_id: Optional[str]
    trace_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = None
    events: List[Dict[str, Any]] = None

@dataclass
class Metric:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    type: str  # counter, gauge, histogram

class ObservabilityContext:
    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.span_id = str(uuid.uuid4())
        self.parent_span_id = None
        self.attributes: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
        self.spans: List[Span] = []
        self.metrics: List[Metric] = []
        self.data_lineage: Dict[str, List[str]] = {}

# Global context variable
observability_context: ContextVar[ObservabilityContext] = ContextVar('observability_context')

class ObservabilityTracer:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    def start_span(self, name: str, attributes: Dict[str, Any] = None) -> Span:
        ctx = observability_context.get()
        span = Span(
            span_id=str(uuid.uuid4()),
            parent_span_id=ctx.span_id,
            trace_id=ctx.trace_id,
            name=name,
            start_time=datetime.utcnow(),
            attributes=attributes or {},
            events=[]
        )
        ctx.spans.append(span)
        return span

    def end_span(self, span: Span):
        span.end_time = datetime.utcnow()
        self._export_span(span)

    def add_event(self, span: Span, name: str, attributes: Dict[str, Any] = None):
        event = {
            'name': name,
            'timestamp': datetime.utcnow().isoformat(),
            'attributes': attributes or {}
        }
        span.events.append(event)

    def _export_span(self, span: Span):
        span_data = asdict(span)
        span_data['start_time'] = span_data['start_time'].isoformat()
        span_data['end_time'] = span_data['end_time'].isoformat() if span_data['end_time'] else None
        if span_data['events']:
            for event in span_data['events']:
                if 'timestamp' in event:
                    event['timestamp'] = event['timestamp'].isoformat()
        self.logger.info(f"Span completed: {json.dumps(span_data)}")

class MetricsCollector:
    def __init__(self):
        self.logger = logging.getLogger('metrics')

    def record_metric(self, name: str, value: float, metric_type: str, labels: Dict[str, str] = None):
        ctx = observability_context.get()
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            labels={**(labels or {}), 'trace_id': ctx.trace_id},
            type=metric_type
        )
        ctx.metrics.append(metric)
        self._export_metric(metric)

    def _export_metric(self, metric: Metric):
        metric_data = asdict(metric)
        metric_data['timestamp'] = metric_data['timestamp'].isoformat()
        self.logger.info(f"Metric recorded: {json.dumps(metric_data)}")

class DataLineageTracker:
    def track_data_flow(self, source: str, destination: str, operation: str = None):
        ctx = observability_context.get()
        if source not in ctx.data_lineage:
            ctx.data_lineage[source] = []
        ctx.data_lineage[source].append({
            'destination': destination,
            'operation': operation,
            'timestamp': datetime.utcnow().isoformat(),
            'trace_id': ctx.trace_id
        })

def with_observability(func):
    """Decorator to add observability context to functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = ObservabilityContext()
        token = observability_context.set(ctx)
        
        tracer = ObservabilityTracer(__name__)
        metrics = MetricsCollector()
        lineage = DataLineageTracker()

        span = tracer.start_span(
            name=func.__name__,
            attributes={
                'args': str(args),
                'kwargs': str(kwargs)
            }
        )

        try:
            start_time = datetime.utcnow()
            result = await func(*args, **kwargs)
            
            # Record execution metrics
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            metrics.record_metric(
                name=f"{func.__name__}_duration_ms",
                value=duration,
                metric_type='histogram',
                labels={'status': 'success'}
            )

            # Track data lineage if result contains data
            if isinstance(result, dict) and 'data' in result:
                lineage.track_data_flow(
                    source=func.__name__,
                    destination='client',
                    operation='process_and_return'
                )

            tracer.add_event(span, 'function_completed', {
                'duration_ms': duration,
                'status': 'success'
            })
            
            return result

        except Exception as e:
            metrics.record_metric(
                name=f"{func.__name__}_errors",
                value=1,
                metric_type='counter',
                labels={'error_type': type(e).__name__}
            )

            tracer.add_event(span, 'function_error', {
                'error_type': type(e).__name__,
                'error_message': str(e)
            })
            raise

        finally:
            tracer.end_span(span)
            observability_context.reset(token)

    return wrapper 