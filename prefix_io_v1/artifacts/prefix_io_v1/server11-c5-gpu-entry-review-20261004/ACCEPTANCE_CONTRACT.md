# C5 GPU entry revision: independent CPU boundary review

This review addresses only the new native entry revision, not the 65 earlier
combined-entry tests or the 132-trial CPU comparison. Its CPU tests must not
import Torch, vLLM, py-kvcache native backends, load a model, reserve GPU budget,
create an effective GPU configuration/scope, or issue a native cost receipt.

The review checks real new entry functions and their rejection boundaries:
missing binding, unresolved or changed UUID, absent permission, mismatched
label/source/job/guard, source drift, old grants and explicitly synthetic raw
data. A test that merely calls the old unconditional CPU block is insufficient
evidence of the new boundary. Rejection must occur before GPU/model access.

The canonical receipt remains the exact class imported by the unchanged policy
and bridge. Dict/PASS values, subclass substitutes, CPU fixture capture, origin
relabeling and private constructor/bridge mutation cannot stand in for real
native evidence. No positive on installation or real GPU qualification can be
claimed by this review.

The original model/cache pipeline, AB/BA/AB six-window design, 129 prompt / 128
cached / 128 output tokens, measured offset 16, 917504-byte single SSD read,
eight accepted parents, calibration formula and independent holdout remain
unchanged. Common calibration has bridge=None. No workload, sample-selection,
margin, budget, formula, deadline or threshold adjustment is permitted here.

Review fixtures are explicitly synthetic rejection inputs, not authorized
configuration, grants or experimental raw data. Every new review execution
records its source identities before and after and keeps all failures. Missing
actual resource or authority is an honest blocker, not a reason to fake a
receipt. Report native/GPU/on/full-entry/performance qualification as false.
