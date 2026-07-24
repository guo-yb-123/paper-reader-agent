"""
Critic Agent - 批判性审阅专家
- 输入：三个 Reader 的全部总结
- 输出：批判性分析清单（创新性评估、实验充分性质疑、局限性分析、潜在问题）
- 角色：严格的学术审稿人，站在批判角度挑问题
"""
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.config import get_llm, invoke_llm


CRITIC_SYSTEM_PROMPT = """你是一位严格的学术论文审稿人（Reviewer），拥有多年的顶会/顶刊审稿经验。

你的任务是基于三个专业 Reader Agent 的分析报告，对论文进行批判性审阅。你不应该简单复述论文内容，而是站在批评和质疑的角度，找出论文的潜在问题。

## 审阅维度

### 1. 创新性评估
- 论文的核心创新点是否真正新颖？还是在已有工作上的微小改进？
- 创新的技术含量如何？是否属于"换皮"式创新？
- 与前人工作的本质区别在哪里？是否被过度夸大？
- 论文对领域的贡献是增量式的还是突破式的？

### 2. 方法论审视
- 方法设计是否存在逻辑漏洞或理论缺陷？
- 是否有未声明的假设或限制条件？
- 方法的通用性如何？是否过度拟合特定场景？
- 算法复杂度是否合理？是否存在效率问题？

### 3. 实验充分性质疑
- 实验设计是否存在缺陷？（数据集选择、划分方式、评估指标等）
- 基线对比是否公平？是否遗漏了重要的基线方法？
- 消融实验是否充分？有没有"选择性消融"的嫌疑？
- 统计显著性检验是否充分？结果波动性如何？
- 是否可能存在数据泄露、过拟合或其他实验偏差？

### 4. 声明与证据匹配度
- 论文的核心声明是否有充分的实验证据支撑？
- 是否存在夸大结论或过度推广的情况？
- 实验结果的提升幅度是否具有实际意义（而不仅是统计意义）？

### 5. 局限性与潜在问题
- 论文未提及的局限性有哪些？
- 方法在哪些场景下可能失效？
- 可复现性如何？是否提供了足够的实现细节？
- 是否存在伦理或公平性方面的隐患？

## 输出格式
请输出结构化的批判性审阅报告，使用 Markdown 格式。
每个发现的问题请标注严重程度：🔴高风险 / 🟡中风险 / 🟢低风险
最后给出总体评分（1-10分）和是否推荐接收的意见。"""


def run_critic(
    background_summary: str,
    method_summary: str,
    experiment_summary: str,
    paper_title: str = "",
) -> str:
    """
    Critic Agent：批判性审阅三个 Reader 的分析结果。

    Args:
        background_summary: Background Reader 的分析
        method_summary: Method Reader 的分析
        experiment_summary: Experiment Reader 的分析
        paper_title: 论文标题

    Returns:
        批判性审阅报告（Markdown 格式）
    """
    logger.info("Critic Agent 开始批判性审阅...")

    # 控制输入长度，防止超出 token 限制
    max_len = 4000
    bg_text = background_summary[:max_len] + ("...(截断)" if len(background_summary) > max_len else "")
    method_text = method_summary[:max_len] + ("...(截断)" if len(method_summary) > max_len else "")
    exp_text = experiment_summary[:max_len] + ("...(截断)" if len(experiment_summary) > max_len else "")

    user_message = f"""请对以下论文的三个维度分析报告进行批判性审阅：

## 论文标题
{paper_title or "（未提供）"}

## Background Reader 分析报告（研究背景与相关工作）
{bg_text}

## Method Reader 分析报告（方法论与技术细节）
{method_text}

## Experiment Reader 分析报告（实验设计与结果）
{exp_text}

请根据以上三份分析报告，按照系统提示中的五个维度，给出你的批判性审阅意见。
重点找出三个 Reader 可能遗漏的问题、矛盾之处，以及论文本身的潜在缺陷。"""

    try:
        llm = get_llm(temperature=0.1)
        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = invoke_llm(llm, messages)
        logger.success(f"Critic Agent 审阅完成，输出 {len(response.content)} 字符")
        return response.content

    except Exception as e:
        logger.error(f"Critic Agent 执行失败: {e}")
        return f"## 批判性审阅失败\n\n错误: {e}"
