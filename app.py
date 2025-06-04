import os
from flask import Flask, request, jsonify, render_template, session
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# 환경변수에서 API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# LangChain LLM 설정
chatgpt = ChatOpenAI(
    model_name='gpt-4o-mini',
    streaming=False,
    temperature=0,
    openai_api_key=openai_api_key
)

app = Flask(__name__)
app.secret_key = "your-secret-key"  # 세션 사용을 위한 비밀 키

# 시스템 프롬프트 설정
system_prompt = SystemMessage(content="""너는 법률 전문 변호사야.
사용자의 질문을 기반으로 최대한 간단하게 아래 템플릿을 바탕으로 적어줘

1. 메시지를 입력한 사용자의 쿼리를 다시한번 확인
2. 법적 절차나 조치
3. 추가적인 정확하고 명백한 정보 과거 판례를 바탕으로 짧게 요약
4. 과거 판례를 기반으로 통계적 형량 예측
5. 최대한 간결 명료하게
6. \\n\\n과 같은 컴퓨터 용어 사용 금지
7. 확신 하지 말 것
8. 정확한 법제처에 있는 과거 판례를 가장 마지막에 정확한 정보로
9. 최대한 문단화 시키기 문단 내에는 최대 2줄 다만 과거 판례에 대한 출처 기반의 텍스트는 제외
10. 명백히 사이버 통신법 위반 사항에 가장 관련도가 있는 정보를 설명

📌너는 아래 내용을 메모리 업데이트 해줘. 답변할 때 chain of thought 방식으로 답변해줘. 단, 사용자의 간단한 질문은 추론 없이 답변만 해도 돼. 유연하게 답변해줘.

...(중략: system_prompt 내용 동일하게 유지)...

핵심 요약 질문을 받으면, 분석 → 실무 → 평가 → 최종 정리 과정을 거친다. 명확성, 정확성, 맥락성을 최우선으로 고려하고, 필요 시 추가 질의를 통해 문제 정의를 명료화한다. 답변 완성 후에도, 사용자가 요청하면 재평가·수정 과정을 거쳐 개선된 답변을 제시한다.""")

@app.route("/")
def home():
    return render_template("index.html")  # index.html이 templates 폴더 안에 있어야 함

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    # 세션 히스토리 초기화
    if "chat_history" not in session:
        session["chat_history"] = []

    # 히스토리를 LangChain 메시지 형식으로 구성
    messages = [system_prompt]
    for msg in session["chat_history"]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # 새로운 질문 추가
    messages.append(HumanMessage(content=question))

    # 응답 생성
    response = chatgpt(messages)
    answer = response.content

    # 세션 업데이트
    session["chat_history"].append({"role": "user", "content": question})
    session["chat_history"].append({"role": "ai", "content": answer})

    return jsonify({"answer": answer})

@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    return jsonify({"message": "Memory reset complete."})

if __name__ == "__main__":
    app.run(debug=True)
