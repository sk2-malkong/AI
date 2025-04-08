from models.model_loader import predict
from utils.gpt_checker import refine_abusive_text

def hybrid_abuse_check(text: str):
    ml_result = predict(text)

    if ml_result["is_abusive"]:
        gpt_result = refine_abusive_text(text)
        return {
            "isAbusive": True,
            "score": ml_result["score"],
            "source": "ML+GPT",
            "abusivePart": gpt_result["abusive_part"],
            "refined": gpt_result["refined"]
        }
    else:
        return {
            "isAbusive": False,
            "score": ml_result["score"],
            "source": "ML",
            "abusivePart": None,
            "refined": text
        }
