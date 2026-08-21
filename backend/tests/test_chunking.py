from app.chunking import chunk_passage, indexable_chunks


def test_produces_diverse_strategies_and_source_metadata():
    chunks = chunk_passage("One short fact is here. Another related fact explains the first one. A third sentence changes subject completely.", "doc-1", "en")
    assert {c.strategy for c in chunks} >= {"parent_passage", "sentence_window_1", "semantic_topic"}
    assert all(c.source_id == "doc-1" and c.parent_text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_index_selection_keeps_multiple_chunking_strategies_under_a_budget():
    chunks = chunk_passage("One fact starts this passage. " * 80, "doc-2", "en")
    indexed = indexable_chunks(chunks)
    assert len(indexed) <= 6
    assert {chunk.strategy for chunk in indexed} >= {"parent_passage", "semantic_topic", "fixed_96_overlap_24"}
