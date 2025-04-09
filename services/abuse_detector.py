from models.model_loader import predict
from models.KoVicuna import load_kovicuna_model, predict_abuse
from utils.gpt_checker import refine_abusive_text

tokenizer, model = load_kovicuna_model()

'''
def hybrid_abuse_check(text: str):
    ml_result = predict(text)
    AI_result = predict_abuse

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
'''

def hybrid_abuse_check(text: str):
    ml_result = predict(text)

    # 1차: ML이 욕설이라고 판단하면 바로 순화
    if ml_result["is_abusive"]:
        gpt_result = refine_abusive_text(text)
        return {
            "isAbusive": gpt_result["gpt_abusive"],
            "part_num": gpt_result["part_num"],
            "score": ml_result["score"],
            "source": "ML+GPT",
            "abusivePart": gpt_result["abusive_part"],
            "refined": gpt_result["refined"],
            "report": gpt_result["report"]
        }

    # 2차: ML이 비욕설로 판단한 경우 → KoVicuna로 재확인
    llm_response = predict_abuse(text, tokenizer, model)

    if "욕설 여부: true" in llm_response:
        gpt_result = refine_abusive_text(text)
        return {
            "isAbusive": gpt_result["gpt_abusive"],
            "part_num": gpt_result["part_num"],
            "score": ml_result["score"],
            "source": "KoVicuna+GPT",
            "abusivePart": gpt_result["abusive_part"],
            "refined": gpt_result["refined"],
            "report": gpt_result["report"]
        }

    # 3차: ML과 KoVicuna 모두 비욕설 → 원문 그대로 반환
    return {
        "isAbusive": False,
        "part_num": "0",
        "score": ml_result["score"],
        "source": "ML+KoVicuna+GPT",
        "abusivePart": None,
        "refined": text,
        "report": None
    }
