from app import classify_ticket


def test_duplicate_charge_is_high_priority():
    result = classify_ticket("I was charged twice for my order.")

    assert result.category == "billing"
    assert result.priority == "high"
    assert result.needs_human_review is True


def test_technical_issue_is_classified():
    result = classify_ticket(
        "The app crashes whenever I try to upload a photo."
    )

    assert result.category == "technical"
    assert result.suggested_team.lower() in ("technical", "engineering")


def test_empty_ticket_is_rejected():
    try:
        classify_ticket("")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty tickets should be rejected")