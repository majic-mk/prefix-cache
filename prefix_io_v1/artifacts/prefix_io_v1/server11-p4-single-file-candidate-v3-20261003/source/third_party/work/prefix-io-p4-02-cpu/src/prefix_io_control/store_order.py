"""P3 strong-simple-baseline experiment: bounded mandatory store precedence.

Reorders existing parent references for one native store pass. Does not queue,
submit, split, complete, reserve, release, query events, or grant ordinary credit.
"""
from itertools import chain,islice
from threading import get_ident

class MandatoryStoreOrder:
    def __init__(self, *, candidate_parents=32):
        if type(candidate_parents) is not int or candidate_parents != 32:
            raise ValueError("P3 freezes the candidate window to 32 native parents")
        self.candidate_parents=candidate_parents
        self.bound=False
        self.owner=None
        self.faulted=False
        self.error=None
        self.calls=self.selected_parents=self.reordered_passes=self.truncated_windows=0

    def bind(self):
        if self.bound:
            raise ValueError("store order already belongs to another reactor")
        self.bound=True

    def fail(self, reason):
        self.faulted=True
        self.error=str(reason)[:160]

    def ordered(self, active, required):
        if not self.bound:
            raise RuntimeError("unbound store-order contract")
        ident=get_ident()
        if self.owner is None:self.owner=ident
        if self.owner!=ident:
            raise RuntimeError("store order must execute on the reactor owner")
        if self.faulted or not required:
            return active
        self.calls+=1
        self.truncated_windows+=int(len(active)>self.candidate_parents)
        candidates=tuple(islice(active,self.candidate_parents))
        selected=tuple(j for j in candidates if j.is_store and j.failed is None
                       and not j.future_set and j.next_file_index<j.total_files
                       and j.future in required)
        if not selected:
            return active
        chosen={id(j) for j in selected}
        self.selected_parents+=len(selected)
        # This is an observed order change, not proof a selected file was issued.
        first_native=next((j for j in candidates if j.is_store and j.failed is None
                           and not j.future_set and j.next_file_index<j.total_files),None)
        self.reordered_passes+=int(first_native is not selected[0])
        # All unselected/native-outside-window jobs remain in the same pass.
        # The generator is consumed immediately by the original native loop.
        return chain(selected,(j for j in active if id(j) not in chosen))

    def snapshot(self):
        """Read after native shutdown; counts describe passes, not unique jobs."""
        return dict(candidate_parents=self.candidate_parents,calls=self.calls,
                    selected_parents=self.selected_parents,reordered_passes=self.reordered_passes,
                    truncated_windows=self.truncated_windows,faulted=self.faulted,error=self.error,
                    scope="bounded mandatory-first parent iteration; no quotas or release credit")
