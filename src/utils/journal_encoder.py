"""
JournalJSONEncoder: an explicit, auditable replacement for the previous
`json.dump(..., default=str)` used to persist a Journal.

WHY THIS EXISTS: `default=str` is only ever invoked for a value the
standard `json` module doesn't already know how to serialize -- and when
it fires, it silently stringifies THAT VALUE, no matter what it was. That
is fine for a value that's only ever read back as text (this project's
own exc_info/exc_stack fields, for example -- see the audit below). It is
NOT fine for a value that downstream code expects to do arithmetic on: if
`Node.metric` ever ended up holding something json can't natively
serialize -- e.g. a `numpy.float64` instead of a plain Python `float`,
which is exactly what `sklearn.metrics.mean_squared_error(...)` returns
if you print/store it without an explicit `float(...)` cast -- `default=
str` would silently turn it into the STRING "1903868428.9" in the JSON
file. `Journal.from_dict()` would then load `metric` back as a Python
`str`, and everything downstream that compares metrics numerically
(`Journal.get_best_node()`'s `min(..., key=lambda n: n.metric)`,
`Agent._node_value()`'s `1.0 / (1.0 + node.metric)`) would either do
silent, wrong lexicographic string comparison or raise a `TypeError`
several files away from the actual cause -- with no warning anywhere
that a value had quietly changed type on its way through a checkpoint
file.

CURRENT STATE, audited directly against this codebase (not assumed): as
of this file's introduction, nothing actually hits the fallback path.
`Node.metric` is always either `None` or a real Python `float` (it comes
from a Pydantic-validated `ExecutionEvaluation.metric: float` field --
see src/agent/schemas.py -- which coerces whatever the LLM's JSON output
said into a native float; the standalone driver scripts that don't go
through that path parse a printed metric with an explicit `float(...)`
call too). `Node.exc_info`/`exc_stack` (src/interpreter/interpreter.py's
`exception_summary()`) are built from `str(...)` calls and plain
tuples/dicts already, so they're natively JSON-safe too. `tests/
test_journal_encoder.py::test_current_node_shapes_never_hit_the_fallback`
asserts this stays true -- if a future change makes it NOT true, that
test fails loudly instead of the type-corruption happening silently.

So this class is deliberately defense-in-depth rather than a fix for an
active bug: it turns an implicit, silent "whatever, stringify it" fallback
into an explicit, narrow, LOGGED one -- exact numeric fixes for the
specific types most likely to sneak in (numpy scalars, Decimal), and a
warning (not a silent pass) for anything else, so a future regression
shows up in your logs the moment it's serialized, not as a mysterious
string-vs-float bug three files away.
"""

import json
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)


class JournalJSONEncoder(json.JSONEncoder):
    """Drop-in for `json.dump(..., cls=JournalJSONEncoder)`."""

    def default(self, obj):
        # numpy scalar types (numpy.float64, numpy.int64, numpy.bool_,
        # ...) all implement .item() to convert back to the equivalent
        # native Python type with no precision loss -- this is the
        # precise fix for the "metric came back from sklearn/numpy, not
        # a plain float" scenario described above.
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                pass

        if isinstance(obj, Decimal):
            return float(obj)

        logger.warning(
            "JournalJSONEncoder: no exact handler for type %s (value=%r); "
            "falling back to str(). If this value is ever compared/used "
            "arithmetically after being reloaded (e.g. it reached a "
            "Node.metric-like field), this WILL silently break after a "
            "round-trip through disk -- add an explicit case above "
            "instead of relying on this fallback.",
            type(obj).__name__,
            obj,
        )
        return str(obj)
