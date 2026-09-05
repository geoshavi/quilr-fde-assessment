"""Pydantic v2 input models for the get_customer_record and trigger_refund tools.

ASSUMPTION (documented in README): CUST-XXXXX means exactly five decimal digits
after the literal prefix "CUST-" (e.g. CUST-12345). Lowercase, extra/missing
digits, non-digit characters, and surrounding whitespace are all rejected.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ASCII digits only. Plain `\d` in a pydantic-core pattern is Unicode-aware (matches
# any Unicode category-Nd digit, e.g. fullwidth "１２..."), which would let a
# visually-similar but distinct customer_id slip past "exactly five digits" (M1 review
# finding, verified empirically: "CUST-" + fullwidth "12345" was accepted before this fix).
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"


def _reject_non_numeric_amount(value: object) -> object:
    """Reject bool/str/None/etc. before pydantic's lax numeric coercion runs.

    Pydantic v2's default lax mode silently coerces numeric strings (e.g. "10.5")
    and bool (True/False are int subclasses) into float. That coercion is
    empirically confirmed (see M1 verification) and is exactly the kind of
    permissive coercion the assessment asks us to avoid for `amount`, so it is
    blocked explicitly here rather than left to the default behavior.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("amount must be an int or float, not a string, bool, or other type")
    return value


def _reject_non_string_reason(value: object) -> object:
    """Reject bytes/etc. before pydantic's lax str coercion runs.

    Pydantic v2's default lax mode decodes `bytes` into `str` for a `str` field.
    Not reachable over the real JSON-RPC wire (JSON has no bytes type), but
    blocked anyway for consistency with the project's stance against permissive
    coercion (M1 review finding), since the model is also called directly in tests.
    """
    if not isinstance(value, str):
        raise ValueError("reason must be a string, not bytes or any other type")
    return value


class GetCustomerRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)


class TriggerRefundInput(BaseModel):
    # allow_inf_nan=False rejects float('inf')/float('-inf')/float('nan') for `amount`
    # at the validation boundary. Without it, `gt=0` alone lets +inf through (inf > 0 is
    # True), which previously reached mock_data.trigger_refund and crashed with an
    # uncaught OverflowError there (M1 review finding).
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    customer_id: str = Field(pattern=CUSTOMER_ID_PATTERN)
    amount: float = Field(gt=0)
    reason: str = Field(min_length=10)

    @field_validator("amount", mode="before")
    @classmethod
    def _validate_amount_type(cls, value: object) -> object:
        return _reject_non_numeric_amount(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _validate_reason_type(cls, value: object) -> object:
        return _reject_non_string_reason(value)
