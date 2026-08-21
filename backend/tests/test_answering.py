from app.answering import grounded_answer
from app.contracts import Citation


def _cit(text: str, score: float = 0.6, sid: str = "doc-1") -> Citation:
    return Citation(source_id=sid, text=text, score=score, strategy="parent_passage")


def test_answers_when_sentence_echoes_question_content_terms():
    q = "What is the capital of France?"
    answer, reason, used = grounded_answer(q, [_cit("The capital of France is Paris, located on the Seine river.")], 0.34)
    assert answer is not None and "Paris" in answer
    assert len(used) == 1


def test_refuses_when_only_stopwords_overlap():
    q = "What is the capital of France?"
    answer, reason, used = grounded_answer(
        q, [_cit("Determining the average profit margin of a small business depends on what qualifies as a small business.", score=0.0)], 0.34)
    assert answer is None
    assert reason is not None


def test_refuses_unrelated_high_dense_score():
    q = "What is the capital of France?"
    answer, reason, used = grounded_answer(
        q, [_cit("Land conquered by Conquistadors. Thank you for visiting our website!", score=0.66)], 0.34)
    assert answer is None


def test_refuses_partial_content_term_overlap():
    q = "What is the capital of France?"
    answer, reason, used = grounded_answer(
        q, [_cit("The cost of capital and the location of a business determine its average profit margin.", score=0.0)], 0.34)
    assert answer is None


def test_lexical_only_citation_can_answer_when_truly_overlapping():
    q = "Where is the Seine river located?"
    answer, reason, used = grounded_answer(
        q, [_cit("The Seine river is located in Paris, the capital of France.", score=0.0)], 0.34)
    assert answer is not None


def test_idf_weights_distinctive_terms_over_generic_ones():
    idf = {"seine": 8.0, "river": 3.0, "located": 1.5}
    q = "Where is the Seine river located?"
    answer, reason, _ = grounded_answer(
        q, [_cit("The river is located between two hills on the west side.", score=0.0)], 0.34, idf)
    assert answer is None
    answer, reason, _ = grounded_answer(
        q, [_cit("The Seine river is located in Paris, France.", score=0.0)], 0.34, idf)
    assert answer is not None


def test_high_dense_score_alone_is_not_grounding_without_distinctive_term():
    idf = {"seine": 8.0, "river": 3.0, "located": 1.5}
    q = "Where is the Seine river located?"
    # score 0.66 above any relevance bar, but shares no distinctive term "seine"
    answer, reason, _ = grounded_answer(
        q, [_cit("Land conquered by Conquistadors. Thank you for visiting!", score=0.66)], 0.34, idf)
    assert answer is None