import gradio as gr
from graph_tools import (
    chat_model,
    retrieve_jd_info,
    analyze_jd_info,
    analyze_and_match_resume,
)
from langchain.agents import create_agent
from gradio import ChatMessage
from langchain_core.messages import AIMessageChunk, ToolMessage, ToolMessageChunk
import json
from langgraph.checkpoint.memory import InMemorySaver


memory=InMemorySaver()
tools=[retrieve_jd_info, analyze_jd_info, analyze_and_match_resume]
agent=create_agent(
    model=chat_model,
    tools=tools,
    system_prompt=("你是招聘助手，有三个工具：retrieve_jd_info 检索公司已有的岗位JD库；analyze_jd_info 分析一段JD原文；"
                   "analyze_and_match_resume 简历一站式工具，输入简历原文即可自动完成候选人画像、岗位库检索和匹配度报告。"
                   "路由规则：1）用户直接给出JD原文，必须调用 analyze_jd_info 做结构化分析；"
                   "2）用户没给JD、只是询问岗位信息，调用 retrieve_jd_info 检索；"
                   "3）用户发来简历、或要求分析候选人/匹配岗位时，调用一次 analyze_and_match_resume 并传入完整简历原文即可，"
                   "画像和匹配都在工具内部按固定顺序完成，不要分多次调用、不要自己分析简历；"
                   "4）检索结果里确实没有，再如实说资料不足。"
                   "当用户让你从多个岗位里推荐或判断最适合哪一个时，只给出匹配度最高的 1 个岗位，并用一两句话说明理由（最关键的匹配点），不要把所有岗位逐个罗列对比，除非用户明确要求看全部岗位的对比。如果用户曾要求“直接说结果”或“简洁回答”，就要一直保持这种风格：先给结论、不铺垫、不展开，即使用户后面的消息里没有重复这条要求，也必须照做，直到用户明确要求“详细展开”。回答简洁。"
                   ),checkpointer=memory
)
def stream_bot(history):
    user_text = history[-1]["content"]
    config = {"configurable": {"thread_id": "zhang-001"}}
    answer = ""          # 当前普通回复的累积文本
    answer_idx = None    # 普通回复气泡在 history 中的下标
    pending_tools = {}   # 工具调用 index -> {name, args, bubble_idx}
    for chunk, _meta in agent.stream(
        {"messages": [("user", user_text)]},
        stream_mode="messages",
        config=config,
    ):
        if isinstance(chunk, AIMessageChunk):
            # 工具调用参数是按分片到的，这里增量拼接
            if chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    idx = tc["index"]
                    if idx not in pending_tools:
                        pending_tools[idx] = {
                            "name": tc.get("name", "") or "",
                            "args": "",
                            "bubble_idx": None,
                        }
                    if tc.get("name"):
                        pending_tools[idx]["name"] = tc["name"]
                    if tc.get("args"):
                        pending_tools[idx]["args"] += tc["args"]
                    info = pending_tools[idx]
                    if info["bubble_idx"] is None:
                        history.append(ChatMessage(
                            role="assistant",
                            content="",
                            metadata={"title": f"🛠️ 调用工具 {info['name'] or '...'}", "status": "pending"},
                        ))
                        info["bubble_idx"] = len(history) - 1
                    args_done = info["args"].lstrip().startswith("{") and info["args"].rstrip().endswith("}")
                    show_args = info["args"]
                    if args_done:
                        try:
                            show_args = json.dumps(
                                json.loads(info["args"]),
                                ensure_ascii=False,
                                indent=2,
                            )
                        except Exception:
                            pass
                    history[info["bubble_idx"]] = ChatMessage(
                        role="assistant",
                        content=show_args[:500],
                        metadata={
                            "title": f"🛠️ 调用工具 {info['name']}",
                            "status": "done" if args_done else "pending",
                        },
                    )
                yield history
            # 正文按 token 增量累积，逐字更新气泡
            if chunk.content:
                answer += chunk.content
                if answer_idx is None:
                    history.append(ChatMessage(role="assistant", content=""))
                    answer_idx = len(history) - 1
                history[answer_idx] = ChatMessage(role="assistant", content=answer)
                yield history
        elif isinstance(chunk, (ToolMessage, ToolMessageChunk)):
            tool_name = getattr(chunk, "name", "未知工具")
            history.append(ChatMessage(
                role="assistant",
                content=str(chunk.content)[:600],
                metadata={
                    "title": f"⚙️ 工具 {tool_name} 返回结果",
                    "status": "done",
                },
            ))
            answer = ""
            answer_idx = None
            yield history
def add_user(user_text,history):
    return" " ,history + [{"role": "user", "content": user_text}]

theme = gr.themes.Soft(
    primary_hue="sky",
    neutral_hue="slate",
    radius_size="md",
    font=["Microsoft YaHei", "PingFang SC", "sans-serif"],
)

lock_light_js = """
function stripDark() {
  document.querySelectorAll('.dark').forEach(function (element) {
    element.classList.remove('dark');
  });
}
stripDark();
new MutationObserver(stripDark).observe(document.documentElement, {
  attributes: true, subtree: true, attributeFilter: ['class']
});
var originalMatchMedia = window.matchMedia.bind(window);
window.matchMedia = function (query) {
  if (query && query.indexOf('prefers-color-scheme') !== -1) {
    return originalMatchMedia('(prefers-color-scheme: light)');
  }
  return originalMatchMedia(query);
};
"""

with gr.Blocks(title="招聘Agent·思考过程可视化") as demo:
    gr.Markdown(
        """
        # 智能招聘助手
        粘贴 **岗位 JD**，我帮你解析成结构化信息；发来 **简历**，我先做候选人画像、再自动从岗位库匹配最合适的岗位；也可以直接问我岗位的薪资、要求、技术栈。
        """
    )
    chatbot=gr.Chatbot(show_label=False)
    with gr.Row():
        msg=gr.Textbox(placeholder="输入问题，回车发送",scale=8,
            show_label=False,)
        send_button=gr.Button("发送",
            variant="primary",
            scale=1,)
    examples = gr.Examples(
        examples=[
            "帮我分析这份简历并匹配岗位：张三，23岁，软件工程本科，会 Python、LangChain、RAG、Chroma，期望实习薪资3K-5K",
            "AI Agent 开发岗位的薪资范围是多少？",
            "Java 岗位要求几年工作经验？",
            "招聘专员主要负责哪些工作？",
        ],
        inputs=msg,
        label="试试这些问题（点击直接提问）",
    )
    msg.submit(add_user,[msg,chatbot],[msg,chatbot],queue=False).then(stream_bot,chatbot,chatbot)
    examples.load_input_event.then(
        add_user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        stream_bot, chatbot, chatbot
    )
    send_button.click(add_user,[msg,chatbot],[msg,chatbot],queue=False).then(stream_bot,chatbot,chatbot)
    gr.Markdown("""
        <div style="text-align:center; color:#94a3b8; font-size:0.85em; margin-top:12px;">
        技术栈：LangChain · LangGraph · Chroma · BGE · Gradio ｜
        开源地址：<a href="https://github.com/Zane96969/recruit-agent" target="_blank">github.com/Zane96969/recruit-agent</a>
        </div>
        """)
   
demo.launch(theme=theme, js=lock_light_js)