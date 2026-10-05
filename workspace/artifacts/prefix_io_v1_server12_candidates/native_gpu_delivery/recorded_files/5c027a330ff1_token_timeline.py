"""Record actual engine output events without inventing per-token timestamps."""
import math

class TokenTimeline:
    def __init__(self):
        self.tokens=[];self.times=[];self.ambiguous_events=[]
    def append(self,cumulative_tokens,engine_timestamp):
        tokens=list(cumulative_tokens)
        if tokens[:len(self.tokens)]!=self.tokens:
            raise ValueError("previous output tokens changed")
        delta=len(tokens)-len(self.tokens)
        if delta<0:raise ValueError("output shrank")
        if not delta:return
        if not isinstance(engine_timestamp,(float,int)) or not math.isfinite(engine_timestamp) or engine_timestamp<=0:
            raise ValueError("missing engine token timestamp")
        if self.times and engine_timestamp<self.times[-1]:
            raise ValueError("engine timestamp moved backwards")
        if delta!=1:
            self.ambiguous_events.append(dict(new_tokens=delta,timestamp=engine_timestamp))
            # Do not spread one chunk timestamp over multiple token positions.
        self.tokens=tokens
        self.times.append(float(engine_timestamp))
    def export(self):
        exact=not self.ambiguous_events and len(self.tokens)==len(self.times)
        return dict(output_tokens=self.tokens,engine_token_timestamps=self.times,
                    timestamp_scope="engine core output event",per_token_complete=exact,
                    ambiguous_events=self.ambiguous_events,
                    itl_seconds=[b-a for a,b in zip(self.times,self.times[1:])] if exact else None)

def percentile(values,p):
    if not values: return None
    values=sorted(values)
    x=(len(values)-1)*p;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]*(1-(x-lo))+values[hi]*(x-lo)
