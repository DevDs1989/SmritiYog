"""Content Agent: turn (game_type, difficulty) into a batch of playable rounds.

Answers are never invented -- for RAG-backed rounds the `correct_answer` comes
straight out of the retrieved item. Gemini is used for exactly one thing, in
exactly one batched call: writing the natural-language `prompt_text` (and the
voice clue) for every slot at once.
"""

import json
import logging
import random

from app import config
from app.agents.state import PatientState
from app.services import gemini
from app.services.retrieval import filter_structured, retrieve_top_k

log = logging.getLogger(__name__)

# --- generated (non-personal) content pools -------------------------------

VISUAL_SETS = [
    {"base": "red_apple", "low": "wooden_chair", "high": "green_apple"},
    {"base": "orange", "low": "bicycle", "high": "tangerine"},
    {"base": "sunflower", "low": "teacup", "high": "marigold"},
    {"base": "sparrow", "low": "hammer", "high": "pigeon"},
    {"base": "banana", "low": "umbrella", "high": "plantain"},
    {"base": "cow", "low": "kettle", "high": "buffalo"},
    {"base": "steel_tumbler", "low": "sandal", "high": "brass_tumbler"},
]

HOUSEHOLD_ITEMS = [
    "table", "chair", "almirah", "clock", "fan", "bucket",
    "pillow", "broom", "mirror", "lamp", "plate", "comb",
]

SOUND_CUES = [
    "doorbell", "temple_bell", "rain", "pressure_cooker_whistle",
    "birdsong", "phone_ring", "footsteps", "water_tap",
]

GENERIC_FAVOURITES = [
    "favourite film", "favourite song", "favourite festival",
    "favourite sweet", "favourite place to visit",
]

RELATION_QUERY = "a close family member described with warm personal details"


def _answer_of(item: dict) -> str:
    meta = item.get("metadata") or {}
    return str(
        meta.get("answer") or meta.get("person_name") or meta.get("name")
        or meta.get("relation") or item.get("text", "")
    )


# --- slot builders: one per game_type -------------------------------------
# Each slot is fully playable already; `brief` is what Gemini gets asked to
# phrase, `fallback_text` is what ships if that call fails.


def _visual_slots(difficulty: str, n: int) -> list[dict]:
    count = 6 if difficulty == "hard" else 4
    level = "low" if difficulty == "easy" else "high"
    slots = []
    for _ in range(n):
        s = random.choice(VISUAL_SETS)
        odd = s[level]
        items = [s["base"]] * (count - 1) + [odd]
        random.shuffle(items)
        idx = items.index(odd)
        slots.append(
            {
                "correct_answer": odd,
                "metadata": {
                    "items": items,
                    "odd_one_out_index": idx,
                    "similarity_level": level,
                },
                "brief": {"task": "odd one out", "items": items},
                "fallback_text": "Which one of these does not belong with the others?",
            }
        )
    return slots


def _favorite_slot(rag_delta: list[dict]) -> dict:
    hits = filter_structured(rag_delta, metadata_filter={"tags": "favourite"})
    if hits:
        item = random.choice(hits)
        return {
            "correct_answer": _answer_of(item),
            "metadata": {"subtype": "favorite_recall"},
            "brief": {"task": "ask about a favourite", "fact": item["text"]},
            "fallback_text": f"Tell me about your {item['text']}.",
        }
    topic = random.choice(GENERIC_FAVOURITES)
    return {
        # Open-ended: no objective answer, the app accepts any confident reply.
        "correct_answer": "",
        "metadata": {"subtype": "favorite_recall"},
        "brief": {"task": "ask about a favourite", "topic": topic},
        "fallback_text": f"What is your {topic}?",
    }


def _memory_voice_slots(difficulty: str, n: int, rag_delta: list[dict]) -> list[dict]:
    if difficulty == "easy":
        return [_favorite_slot(rag_delta) for _ in range(n)]

    if difficulty == "medium":
        slots = []
        for _ in range(n):
            seq = random.sample(HOUSEHOLD_ITEMS, 3)
            slots.append(
                {
                    "correct_answer": ", ".join(seq),
                    "metadata": {"subtype": "sequence_recall", "sequence": seq},
                    "brief": {"task": "read out a list to remember", "sequence": seq},
                    "fallback_text": (
                        "Listen carefully, then repeat these back in order: "
                        + ", ".join(seq)
                    ),
                }
            )
        return slots

    # hard: alternate photo_recall (RAG) and sequence_from_sound (generated).
    # photo_recall is open-ended on purpose -- the patient names the person AND
    # volunteers whatever else they remember, and POST /recall/parse turns that
    # narration into new RAG facts.
    photos = filter_structured(rag_delta, kind="photo")

    slots = []
    for i in range(n):
        if i % 2 == 0 and photos:
            item = random.choice(photos)
            slots.append(
                {
                    "correct_answer": _answer_of(item),
                    "metadata": {"subtype": "photo_recall", "photo_ref": item["id"]},
                    "brief": {
                        "task": "ask who is in this photo and what they remember",
                        "photo": item["text"],
                    },
                    "fallback_text": (
                        "Who is this? Tell me everything you remember about them."
                    ),
                    "used": item,
                }
            )
        elif i % 2 == 0:
            # Nothing retrieved -- generic round beats a forced one.
            slots.append(_favorite_slot(rag_delta))
        else:
            cues = random.sample(SOUND_CUES, 3)
            slots.append(
                {
                    "correct_answer": ", ".join(cues),
                    "metadata": {"subtype": "sequence_from_sound", "audio_cues": cues},
                    "brief": {"task": "play sounds then ask for their order", "cues": cues},
                    "fallback_text": (
                        "You will hear three sounds. Tell me the order you heard them in."
                    ),
                }
            )
    return slots


def _relationship_slots(difficulty: str, n: int, rag_delta: list[dict]) -> list[dict]:
    slots: list[dict] = []

    if difficulty in ("easy", "medium"):
        closeness = "close" if difficulty == "easy" else "extended"
        photos = filter_structured(
            rag_delta, kind="photo", metadata_filter={"closeness": closeness}
        ) or filter_structured(rag_delta, kind="photo")
        for _ in range(n):
            if not photos:
                break  # no non-personalized fallback for this game type
            item = random.choice(photos)
            slots.append(
                {
                    "correct_answer": _answer_of(item),
                    "metadata": {
                        "mode": "guess_person_from_photo",
                        "photo_ref": item["id"],
                    },
                    "brief": {"task": "ask who is in this photo", "photo": item["text"]},
                    "fallback_text": "Who is this in the photo?",
                    "used": item,
                }
            )
        return slots

    # hard: semantic rerank over person notes, then a generated voice clue
    people = filter_structured(rag_delta, kind="person")
    ranked = retrieve_top_k(
        _query_embedding(rag_delta), people, k=max(3, n // 2)
    ) or people
    for i in range(n):
        if not ranked:
            break
        item = ranked[i % len(ranked)]
        meta = item.get("metadata") or {}
        notes = meta.get("notes") or item["text"]
        slots.append(
            {
                "correct_answer": str(meta.get("relation") or _answer_of(item)),
                "metadata": {"mode": "guess_relation_from_voice", "voice_clue_text": None},
                "brief": {"task": "write a first-person voice clue", "notes": notes},
                "fallback_text": f"I am the one who {notes}. Who am I to you?",
                "needs_clue": True,
                "used": item,
            }
        )
    return slots


# --- Gemini: one batched call for all the wording -------------------------

_WRITE_PROMPT = """You write spoken prompts for a memory game used by elderly
dementia patients in India. Voice is warm, calm, second person, one short
sentence, simple words. Never reveal the answer.

For each brief below, write the prompt the app will speak. If a brief has
"task": "write a first-person voice clue", also write a 1-2 sentence clue
spoken AS that family member without naming themselves or their relation.

Return ONLY a JSON array of the same length, each element:
{{"prompt_text": "...", "voice_clue_text": "..." or null}}

Briefs:
{briefs}"""


def _write_texts(slots: list[dict]) -> None:
    """Fill prompt_text / voice_clue_text on each slot, in place."""
    for s in slots:
        s["prompt_text"] = s["fallback_text"]

    if not slots:
        return
    try:
        raw = gemini.generate_text(
            _WRITE_PROMPT.format(briefs=json.dumps([s["brief"] for s in slots]))
        )
        written = gemini.parse_json(raw)
    except Exception:
        log.warning("prompt wording call failed; using templates", exc_info=True)
        written = []  # templates already in place

    for slot, w in zip(slots, written):
        if isinstance(w, dict) and w.get("prompt_text"):
            slot["prompt_text"] = w["prompt_text"]
        if slot.get("needs_clue"):
            clue = (w or {}).get("voice_clue_text") if isinstance(w, dict) else None
            slot["metadata"]["voice_clue_text"] = clue or slot["fallback_text"]

    for slot in slots:
        if slot.get("needs_clue") and not slot["metadata"].get("voice_clue_text"):
            slot["metadata"]["voice_clue_text"] = slot["fallback_text"]


def _query_embedding(rag_delta: list[dict]) -> list[float]:
    return next(
        (i["embedding"] for i in rag_delta if i.get("id") == "__query__"), []
    )


def content_node(state: PatientState) -> dict:
    rag_delta = [dict(i) for i in state.get("rag_delta") or []]

    # One embedding call for the whole delta plus the rerank query.
    embeddings: list[dict] = []
    if rag_delta:
        try:
            vectors = gemini.embed_batch([i["text"] for i in rag_delta] + [RELATION_QUERY])
        except Exception:
            log.warning("embed_batch failed; returning no embeddings", exc_info=True)
            vectors = []
        for item, vec in zip(rag_delta, vectors):
            item["embedding"] = vec
            embeddings.append({"id": item["id"], "embedding": vec})
        if len(vectors) > len(rag_delta):
            rag_delta.append({"id": "__query__", "embedding": vectors[-1]})

    game_type = state["game_type"]
    difficulty = state["difficulty"]
    n = config.BATCH_SIZE

    if game_type == "cognitive_visual":
        slots = _visual_slots(difficulty, n)
    elif game_type == "memory_voice":
        slots = _memory_voice_slots(difficulty, n, rag_delta)
    else:
        slots = _relationship_slots(difficulty, n, rag_delta)
        if not slots:
            # Nothing personal to work with; ship something playable rather
            # than an empty batch.
            game_type, difficulty = "cognitive_visual", "easy"
            slots = _visual_slots(difficulty, n)

    _write_texts(slots)

    rounds = [
        {
            "game_type": game_type,
            "difficulty": difficulty,
            "prompt_text": s["prompt_text"],
            "correct_answer": s["correct_answer"],
            "metadata": s["metadata"],
        }
        for s in slots
    ]
    context = [
        {k: v for k, v in s["used"].items() if k != "embedding"}
        for s in slots
        if s.get("used")
    ]
    return {
        "generated_rounds": rounds,
        "embeddings": embeddings,
        "retrieved_context": context,
    }
