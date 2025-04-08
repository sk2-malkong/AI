from models.model_loader import predict
from utils.gpt_checker import refine_abusive_text

def hybrid_abuse_check(text: str):
    ml_result = predict(text)

    if ml_result["is_abusive"]:
        gpt_result = refine_abusive_text(text)
        return {
            "isAbusive": gpt_result["gpt_abusive"],
            "part_num": gpt_result["part_num"],
            "score": ml_result["score"],
            "source": "ML+LLM",
            "abusivePart": gpt_result["abusive_part"],
            "refined": gpt_result["refined"],
            "report": gpt_result["report"]
        }
    else:
        return {
            "isAbusive": False,
            "part_num":"0",
            "score": ml_result["score"],
            "source": "ML",
            "abusivePart": None,
            "refined": text,
            "report": None
        }
