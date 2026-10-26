import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "gemini-3.8-flash"
FALLBACK_MODEL = "gemini-3.5-flash"
CUISINES = ["Indian", "Italian", "Mexican"]
DIETS = ["Any", "Vegan", "Jain"]
STORES = ["Edeka", "Aldi", "Rewe", "Indian", "African", "Pakistani"]
STARTER_INGREDIENTS = ["Tomato", "Rice", "Chicken", "Onion", "Oil"]


def api_key() -> str | None:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        return None


@st.cache_resource
def get_client() -> genai.Client:
    return genai.Client(http_options={"timeout": 90_000})


def collect_ingredients(picked: list[str] | None, extra: str) -> list[str]:
    ingredients: list[str] = []
    seen: set[str] = set()
    for name in list(picked or []) + extra.replace("\n", ",").split(","):
        cleaned = name.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            ingredients.append(cleaned)
    return ingredients


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "resource_exhausted" in message or "quota" in message or "429" in message


def _is_auth_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "api key" in message or "unauthenticated" in message or "permission_denied" in message


def _complete(model: str, prompt: str) -> str:
    response = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    return (response.text or "").strip()


def recipe_error_message(exc: Exception) -> tuple[str, str]:
    """Return a short title and explanation for a failed generation."""
    if _is_quota_error(exc):
        return (
            "Daily limit reached",
            "The free daily limit for Gemini recipe requests is used up. Try again in a few hours.",
        )
    if _is_auth_error(exc):
        return (
            "API key rejected",
            "Check GOOGLE_API_KEY in the .env file next to this app.",
        )
    if "Timeout" in type(exc).__name__ or "deadline" in str(exc).casefold():
        return (
            "The model took too long",
            "The request timed out. Try again.",
        )
    return (
        "Could not generate a recipe",
        "The model did not return a recipe. Try again in a moment.",
    )


def generate_recipe(
    ingredients: list[str], cuisine: str, diet: str, store: str, area: str
) -> tuple[str, str]:
    """Generate one short recipe, then suggest where to shop.

    Returns the recipe text and the model that produced it.
    """
    place = area.strip()
    location_rule = (
        f"Suggest one realistic place in {place} to buy the ingredients at a {store} shop. "
        "Name the neighborhood and the shop type. Do not invent a street address."
        if place
        else (
            f"Tell the cook to look for a {store} shop near them. "
            "Do not invent a city or a street address."
        )
    )
    prompt = f"""
    Generate one food recipe using these ingredients: {", ".join(ingredients)}
    The recipe itself should not be more than 90 words.
    Cuisine: {cuisine}
    Diet: {diet}
    Reply in Markdown with a title, the ingredients you use, and short steps.
    After the steps, add a heading "Where to shop".
    Store type: {store}. Supermarkets are Edeka, Aldi, and Rewe.
    Indian, African, and Pakistani mean a grocery of that kind.
    {location_rule}
    """
    primary_error: Exception | None = None
    try:
        text = _complete(MODEL, prompt)
        if text:
            return text, MODEL
        primary_error = RuntimeError("The model returned an empty recipe.")
    except Exception as exc:
        if _is_auth_error(exc):
            raise
        primary_error = exc

    try:
        text = _complete(FALLBACK_MODEL, prompt)
    except Exception:
        raise primary_error from None
    if not text:
        raise primary_error or RuntimeError("The model returned an empty recipe.")
    return text, FALLBACK_MODEL


st.set_page_config(
    page_title="Recipe generator",
    page_icon=":material/skillet:",
    layout="centered",
)
st.title("Recipe generator")
st.caption(
    "One short recipe from what you already have, "
    "with a shop suggestion at the end."
)

if not api_key():
    st.error(
        "Add GOOGLE_API_KEY to the .env file next to this app, "
        "or to .streamlit/secrets.toml.",
        title="Missing API key",
    )
    st.stop()

st.session_state.setdefault("recipe", None)
st.session_state.setdefault("recipe_error", None)
st.session_state.setdefault("recipe_note", None)

with st.form("recipe_form"):
    picked = st.pills(
        "Ingredients",
        STARTER_INGREDIENTS,
        selection_mode="multi",
        default=STARTER_INGREDIENTS,
        width="stretch",
        wrap=True,
    )
    extra = st.text_input(
        "More ingredients",
        placeholder="garlic, ginger, cumin",
        help="Separate extra ingredients with commas.",
    )
    cuisine = st.segmented_control(
        "Cuisine",
        CUISINES,
        default="Indian",
        required=True,
        width="stretch",
    )
    diet = st.segmented_control(
        "Diet",
        DIETS,
        default="Any",
        required=True,
        width="stretch",
    )
    store = st.segmented_control(
        "Store",
        STORES,
        default="Edeka",
        required=True,
        width="stretch",
        wrap=True,
        help="Where to look for the ingredients. The recipe ends with a place to shop.",
    )
    area = st.text_input(
        "Area",
        placeholder="Berlin Neukölln",
        help="Neighborhood or city used for the shop suggestion.",
    )
    submitted = st.form_submit_button(
        "Generate recipe",
        type="primary",
        icon=":material/skillet:",
        width="stretch",
    )

result = st.container(border=True)

if submitted:
    ingredients = collect_ingredients(picked, extra)
    if not ingredients:
        st.session_state.recipe = None
        st.session_state.recipe_note = None
        st.session_state.recipe_error = "Add at least one ingredient."
    else:
        st.session_state.recipe_error = None
        with st.status("Writing your recipe", expanded=True) as status:
            try:
                recipe, model = generate_recipe(ingredients, cuisine, diet, store, area)
                st.session_state.recipe = recipe
                st.session_state.recipe_note = (
                    "Gemini 3.8 Flash has reached its free daily limit, "
                    "so this recipe used Gemini 3.5 Flash."
                    if model != MODEL
                    else None
                )
                status.update(label="Recipe ready", state="complete", expanded=False)
            except Exception as exc:
                st.session_state.recipe = None
                st.session_state.recipe_note = None
                title, detail = recipe_error_message(exc)
                st.session_state.recipe_error = (title, detail)
                status.update(label=title, state="error", expanded=False)

if st.session_state.recipe_error:
    title, detail = (
        st.session_state.recipe_error
        if isinstance(st.session_state.recipe_error, tuple)
        else ("Could not generate a recipe", st.session_state.recipe_error)
    )
    result.error(detail, title=title)
elif st.session_state.recipe:
    if st.session_state.recipe_note:
        result.caption(st.session_state.recipe_note)
    result.markdown(st.session_state.recipe)
else:
    result.info(
        "Choose ingredients, a cuisine, a diet, and a store, then generate a recipe.",
        title="Ready when you are",
    )
