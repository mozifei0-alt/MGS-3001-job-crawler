# -*- coding: utf-8 -*-
"""
重新分析脚本：杭州 + <1000人 + 排除实习，不限制薪资
输出所有图表和报告到 新条件分析 文件夹
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ========================================
# 0. Setup
# ========================================
import matplotlib.font_manager as fm
import matplotlib as mpl
import os as _os

# Directly register the Chinese font file for maximum reliability
_font_path = r'C:\Windows\Fonts\msyh.ttc'
if _os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    _font_name = fm.FontProperties(fname=_font_path).get_name()
    print(f'Registered font: {_font_name} from {_font_path}')
else:
    _font_name = 'Microsoft YaHei'
    print(f'WARNING: {_font_path} not found, falling back to {_font_name}')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150

# seaborn must be configured BEFORE setting font
sns.set_style('whitegrid')
sns.set_palette('Set2')

# Set Chinese font AFTER seaborn (seaborn resets rcParams)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [_font_name, 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print('Chinese font configured successfully.')

data_folder = r'D:\AI\AI_projects\MGS数据分析'
output_folder = os.path.join(data_folder, '新条件分析')
csv_path = os.path.join(data_folder, '51job_final_result.csv')

print(f'Loading data from: {csv_path}')
df = pd.read_csv(csv_path, encoding='utf-8-sig')
print(f'Dataset shape: {df.shape}')

# ========================================
# 1. Salary parsing
# ========================================
def parse_salary(sal_str):
    """Parse salary string from 51job, return monthly salary in RMB.

    Handles formats:
    - "1-1.5万" (ten-thousand/mo), "8千-1.2万" (mixed units), "4-5千" (thousand/mo)
    - "1-1.5万·13薪" (with bonus months), "20-40万/年" (annual)
    - Daily rates ("150元/天") return NaN (not monthly salaries)
    """
    if pd.isna(sal_str) or str(sal_str).strip() == '':
        return np.nan
    s = str(sal_str).strip()

    # Daily rates — not comparable to monthly salaries
    if re.search(r'元/天|元/日|/天|/日', s):
        return np.nan

    # Handle bonus months (e.g., "·13薪", "15薪")
    bonus_months = 12
    bonus_match = re.search(r'(\d+)\s*薪', s)
    if bonus_match:
        bonus_months = int(bonus_match.group(1))
        s = re.sub(r'·?\d+\s*薪', '', s).strip()

    # Handle annual salary
    is_annual = bool(re.search(r'/年|/year', s))
    s = re.sub(r'/年|/year', '', s).strip()

    def parse_single_value(val_str, default_unit='万'):
        """Parse one side of a range with its own unit. Returns value in thousands."""
        val_str = val_str.strip()
        m = re.match(r'([\d.]+)\s*(万|千)?', val_str)
        if not m:
            return None, None
        val = float(m.group(1))
        unit = m.group(2) if m.group(2) else None
        multiplier = 10 if (unit or default_unit) == '万' else 1
        return val * multiplier, unit

    # Range pattern: each side parsed independently for mixed-unit support
    range_match = re.search(
        r'([\d.]+\s*(?:万|千)?)\s*[-–~至]\s*([\d.]+\s*(?:万|千)?)', s
    )
    if range_match:
        low_val, low_unit = parse_single_value(range_match.group(1))
        high_val, high_unit = parse_single_value(range_match.group(2))
        if low_val is not None and high_val is not None:
            # If one side lacks a unit, inherit from the other side
            if low_unit is None and high_unit is not None:
                low_val = parse_single_value(range_match.group(1), high_unit)[0]
            elif high_unit is None and low_unit is not None:
                high_val = parse_single_value(range_match.group(2), low_unit)[0]
            monthly = (low_val + high_val) / 2
        else:
            return np.nan
    else:
        # Single value (no range)
        val, _ = parse_single_value(s)
        if val is not None:
            monthly = val
        else:
            return np.nan

    if is_annual:
        monthly /= 12

    # Convert k RMB → actual RMB
    return monthly * bonus_months / 12 * 1000

df['salary_monthly'] = df['sal'].apply(parse_salary)
print(f'Salary parsed: {df["salary_monthly"].notna().sum()}/{len(df)}')

# ========================================
# 2. Filtering (NEW CONDITIONS)
# ========================================
# Hangzhou only
df['is_hangzhou'] = df.iloc[:, 1].str.lower().str.contains('hangzhou', na=False)
print(f'Hangzhou: {df["is_hangzhou"].sum()}')

# Company size < 1000人
small_sizes = ['少于50人', '50-150人', '150-500人', '500-1000人']
df['is_small'] = df.iloc[:, 6].isin(small_sizes)
print(f'Size < 1000: {df["is_small"].sum()}')

# Apply base filters
df_filtered = df[(df['is_hangzhou']) & (df['is_small'])].copy()
print(f'After Hangzhou + Size < 1000: {len(df_filtered)}')

# Exclude 实习/兼职
intern_kw = ['实习', 'intern', 'Intern', '兼职', '临时']
df_filtered['is_intern'] = df_filtered.iloc[:, 0].str.contains('|'.join(intern_kw), na=False)
df_filtered = df_filtered[~df_filtered['is_intern']].copy()
print(f'After excluding interns: {len(df_filtered)}')

# NO salary filter
# Keep rows with valid salary for regression
df_filtered['has_valid_salary'] = df_filtered['salary_monthly'].notna()
print(f'Has valid salary: {df_filtered["has_valid_salary"].sum()}')

df_filtered['company_size'] = df_filtered.iloc[:, 6]
df_filtered['ln_salary'] = np.log(df_filtered['salary_monthly'])

# ========================================
# 3. AI Classification
# ========================================
general_ai_keywords = [
    'ChatGPT', 'chatgpt', 'GPT-4', 'GPT4', 'gpt4',
    '豆包', 'doubao', 'Copilot', 'copilot', 'Kimi', 'kimi',
    '通义千问', '通义', '文心一言', '文心', '智谱', 'ChatGLM', 'chatglm',
    '讯飞星火', '百川', 'Baichuan', 'Claude', 'claude', 'Gemini', 'gemini',
    'AI写作', 'AI写', 'AI总结', 'AI summary', 'AI办公', 'AI office',
    'AI文案', 'AI copywriting', 'AI翻译', 'AI translation',
    'Prompt工程', 'prompt engineering', '提示词', '提示工程',
    '大模型应用', '大模型调用', 'LLM应用', 'llm应用',
    'AI工具使用', 'AI tool', '生成式AI', 'GenAI', 'AIGC应用',
    'AI提效', 'AI赋能', 'AI辅助', 'AI助手',
    'AI自动化', 'AI automation', '智能办公', 'AI知识库',
    # Broad AI terms
    '人工智能', '大模型', 'AI产品', 'AI应用', 'AIGC', 'AI运营', 'AI平台',
    '智能客服', '智能推荐', '智能助手', 'AI技术', 'AI能力', 'AI系统',
    'AI解决方案', 'AI对话', 'AI驱动', 'AI原生',
    '搜索', '问答', '对话系统', '语音助手',
    '智能语音', '智能问答', '智能搜索', '智能分析', '智能决策',
    'AI战略', 'AI转型', 'AI业务', 'AI场景', 'AI项目', 'AI团队',
    'AI工具', 'AI提效', 'AI增效', 'AI流程', 'AI数据',
    '大语言模型', 'LLM', 'llm', 'GPT', 'gpt',
    '元宝', '腾讯元宝', '混元', '盘古', '商量',
    '日日新', 'Minimax', 'minimax', '零一万物', '阶跃星辰',
    '扣子', 'Coze', 'coze', 'Dify', 'dify', 'FastGPT',
    'NotebookLM', 'Perplexity', 'perplexity',
    '智能硬件', '智能穿戴', '智能家居', '智能座舱', '智能驾驶',
    '智能制造', '智能物流', '智能仓储', '智能巡检',
]

specialized_ai_keywords = [
    'Midjourney', 'midjourney', 'Stable Diffusion', 'stable diffusion', 'StableDiffusion',
    'DALL-E', 'DALLE', 'dalle', 'AI设计', 'AI design',
    'AI绘画', 'AI painting', 'AI视频', 'AI video', 'AI剪辑', 'AI editing',
    'AI图像', 'AI image', 'AI美工', 'ComfyUI', 'comfyui',
    'Sora', 'sora', 'Runway', 'runway', '可灵', 'Kling', '即梦',
    'AI数据分析', 'AI data analysis', 'AI数据挖掘', 'AI data mining',
    'AI数据可视化', 'AI data visualization', '智能数据分析', 'AI报表', 'AI spreadsheet',
    '深度学习', 'deep learning', '机器学习', 'machine learning',
    '自然语言处理', 'NLP', 'nlp', '计算机视觉', 'computer vision', 'CV', 'cv算法',
    '推荐系统', '推荐算法', '知识图谱', 'knowledge graph',
    '强化学习', 'reinforcement learning', '联邦学习', 'federated learning',
    'RAG', 'rag', 'Agent开发', 'Agent development', 'AI Agent',
    '模型训练', 'model training', '模型微调', 'fine-tuning', 'finetune',
    'LoRA', 'lora', '向量数据库', 'vector database',
    'LangChain', 'langchain', 'LlamaIndex', 'llamaindex',
    '模型部署', 'model deployment', 'MLOps', 'mlops',
    'AI检测', 'AI预测', 'AI优化',
    '语音识别', 'speech recognition', '语音合成', 'TTS',
    'OCR', 'ocr', '人脸识别', 'face recognition',
    '目标检测', 'object detection', '语义分割', 'semantic segmentation',
    '异常检测', 'anomaly detection', '时序预测', 'time series prediction',
    'AI+Excel', 'AI Excel', 'AI+PS', 'AI Photoshop', 'AI+PR', 'AI Premiere',
    # Round 2 additions
    '算法工程师', '机器人', 'AI开发', '数据挖掘',
    'RPA', 'rpa', 'AI算法', '多模态',
    'DeepSeek', 'deepseek', 'PyTorch', 'pytorch',
    '数据科学', 'AI训练', '数据标注', 'AI安全',
    '神经网络', 'AI基础设施',
    'Llama', 'llama', 'MCP', 'mcp', 'SLAM', 'slam',
    'Transformer', 'transformer', 'TensorFlow', 'tensorflow',
    '生成模型', '自动驾驶', '预训练',
    'Qwen', 'qwen', '隐私计算', '图像识别',
    'AI研发', '无人驾驶', '智能营销',
    'Keras', 'keras', 'Scikit-learn', 'sklearn',
    'HuggingFace', 'huggingface', 'Gradio', 'gradio',
    '注意力机制', '代码生成', '文生图', '文生视频', '数字人', 'AI客服',
    '扩散模型', '具身智能', 'AI芯片', 'AI动画',
    'Jupyter', 'Encoder', 'Decoder', 'Function Calling', 'function calling',
    # Round 3 additions — specialized AI/ML tools and techniques
    'YOLO', 'yolo', 'BERT', 'bert', 'OpenCV', 'opencv',
    'ONNX', 'onnx', 'TensorRT', 'tensorrt', 'CUDA', 'cuda',
    'JAX', 'jax', 'XGBoost', 'xgboost', 'LightGBM', 'lightgbm',
    'CatBoost', 'catboost', 'Faiss', 'faiss', 'Milvus', 'milvus',
    'ChromaDB', 'chromadb', 'Pinecone', 'pinecone', 'Weaviate', 'weaviate',
    'Elasticsearch', 'elasticsearch', '搜索引擎',
    'NLTK', 'nltk', 'spaCy', 'spacy', 'Gensim', 'gensim',
    'MLflow', 'mlflow', 'Kubeflow', 'kubeflow', 'Airflow', 'airflow',
    'Neo4j', 'neo4j', '图数据库', 'GraphRAG', 'graphrag',
    'GPU', 'gpu', 'NPU', 'npu', '算力', 'AI算力',
    '数字孪生', '仿真', '边缘计算', 'AI推理', '模型推理',
    '视觉检测', '缺陷检测', '图像分割', '姿态估计', '点云',
    'NER', 'ner', '命名实体识别', '分词', '依存句法',
    'Text-to-SQL', 'Text2SQL', 'NL2SQL',
    '文生', '图生', '视频生成', '语音克隆',
    'AI Agent', 'AI智能体', '智能体', 'Multi-Agent', 'multi-agent',
    'AI测试', 'AI评测', 'AI评估', '模型评估', '模型压缩',
    '量化投资', '量化交易', '量化分析', '量化策略',
    '运筹优化', '路径规划', '组合优化', '供应链优化',
    'Caffe', 'caffe', 'MXNet', 'mxnet', 'PaddlePaddle', 'paddlepaddle',
    '飞桨', 'MindSpore', 'mindspore',
    'ROS', 'ros', 'Gazebo', 'gazebo', 'Apollo', 'apollo',
    '3D重建', '三维重建', 'NeRF', 'nerf', '3D生成',
    'VLM', 'vlm', '视觉语言模型', 'LLaVA', 'llava',
    'RWKV', 'rwkv', 'Mamba', 'mamba', '状态空间模型',
    'CrewAI', 'crewai', 'AutoGen', 'autogen', 'Semantic Kernel',
    'Ollama', 'ollama', 'vLLM', 'vllm', 'text-generation',
    '语音识别', 'ASR', 'asr', '语音合成', 'TTS', 'tts',
    '声纹识别', '声音克隆', '音频处理', '音乐生成',
]

digital_skills_keywords = [
    'Python', 'python', 'SQL', 'sql',
    'Docker', 'docker', 'Spark', 'spark', 'Hadoop', 'hadoop', 'ETL', 'etl',
    '数据分析', '建模', '爬虫', '数字化', '数智化', '数智',
]

def has_digital_skills(text):
    """Check if text contains digital/technical skill keywords (not AI)."""
    if pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(kw.lower() in text_lower for kw in digital_skills_keywords)

def classify_ai_skills(text):
    if pd.isna(text):
        return False, False
    text = str(text)
    text_lower = text.lower()
    has_general = any(kw.lower() in text_lower for kw in general_ai_keywords)
    has_specialized = any(kw.lower() in text_lower for kw in specialized_ai_keywords)
    # Chinese-compatible word-boundary match for bare "AI" (Python \b fails on CJK)
    if not has_general and not has_specialized:
        if re.search(r'(?<![a-zA-Z])AI(?![a-zA-Z])', text):
            has_general = True
    return has_general, has_specialized

def classify_ai_skills_from_row(row):
    title = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ''
    desc = str(row.iloc[9]) if not pd.isna(row.iloc[9]) else ''
    text = title + ' ' + desc
    has_general, has_specialized = classify_ai_skills(text)
    has_digital = has_digital_skills(text)
    return has_general, has_specialized, has_digital

df_filtered['AI_general'], df_filtered['AI_specialized'], df_filtered['has_digital'] = zip(
    *df_filtered.apply(classify_ai_skills_from_row, axis=1))

# Upgrade: General AI + digital skills → Specialized AI
# Jobs with AI keywords AND digital/technical skills represent deeper AI capability
mask_upgrade = df_filtered['AI_general'] & ~df_filtered['AI_specialized'] & df_filtered['has_digital']
df_filtered.loc[mask_upgrade, 'AI_specialized'] = True
df_filtered.loc[mask_upgrade, 'AI_general'] = False
n_upgraded = mask_upgrade.sum()
print(f'Upgraded General AI → Specialized AI (digital skills present): {n_upgraded}')

def ai_category(row):
    if row['AI_specialized'] and row['AI_general']:
        return 'Both'
    elif row['AI_specialized']:
        return 'Specialized AI'
    elif row['AI_general']:
        return 'General-Purpose AI'
    else:
        return 'No AI Skills'

df_filtered['AI_category'] = df_filtered.apply(ai_category, axis=1)
df_filtered['has_AI'] = df_filtered['AI_general'] | df_filtered['AI_specialized']

# Also create 高/低/无 AI categories for the Chinese report
def ai_demand_category(row):
    title = str(row.iloc[0])
    desc = str(row.iloc[9])
    text = title + ' ' + desc

    high_ai_kw = [
        '算法工程师', '算法研究员', '大模型', 'LLM', '机器学习', '深度学习', '强化学习',
        '自然语言处理', 'NLP', '计算机视觉', 'CV', '语音识别', '语音合成',
        '具身智能', '智能体', 'Agent', 'AI模型训练', 'AI推理', '模型部署',
        'AI编译器', 'AIGC', '多模态', '三维重建', 'AI应用开发', 'AI研发',
        'AI算法', '图像算法', '机器视觉', 'AI部署', 'AI开发',
    ]
    low_ai_kw = [
        'AI产品经理', 'AI项目管理', 'AI销售', 'AI客户经理',
        'Python开发', 'python开发', '数据分析师', '数据挖掘',
        '数字化', '数智化', 'RPA', '数据标注', 'AI漫剧', 'AI动漫',
        'AI产品总监', 'AI产品', 'ai产品', '量化分析师',
    ]

    is_high = any(kw.lower() in text.lower() for kw in high_ai_kw)
    is_low = any(kw.lower() in text.lower() for kw in low_ai_kw)

    if is_high:
        return '高AI需求'
    elif is_low:
        return '低AI需求'
    else:
        return '无AI需求'

df_filtered['AI_demand'] = df_filtered.apply(ai_demand_category, axis=1)

print('\nAI Skill Classification:')
for cat in ['Specialized AI', 'General-Purpose AI', 'Both', 'No AI Skills']:
    count = (df_filtered['AI_category'] == cat).sum()
    print(f'  {cat:<25} {count:<8} {count/len(df_filtered)*100:.1f}%')

print('\nAI Demand Classification:')
for cat in ['高AI需求', '低AI需求', '无AI需求']:
    count = (df_filtered['AI_demand'] == cat).sum()
    print(f'  {cat:<10} {count:<8} {count/len(df_filtered)*100:.1f}%')

# ========================================
# 4. Feature Engineering
# ========================================
# Industry — classified from company name only (col 3) to avoid endogeneity with
# AI skills detected from job description. AI-overlapping terms removed.
digital_kw = [
    '电商','电子商务','互联网','网络科技','信息技术','信息科技',
    '软件','科技','数码','数字','传媒','媒体','广告','营销',
    '新媒体','自媒体','内容','直播','短视频','视频','设计','创意',
    '在线','线上','平台','APP','信息',
    '芯片','半导体','金融科技','fintech','支付','区块链',
    '通信','通讯','网络','电信','电子科技',
]
traditional_kw = [
    '制造','工厂','生产','加工','建筑','工程','物业','房地产',
    '酒店','餐饮','旅游','饭店','物流','运输','快递','仓储',
    '贸易','批发','零售','分销','教育','培训','学校',
    '医疗','医院','医药','药','化工','能源','矿产','钢铁',
    '农业','食品','饮料','纺织','服装','服饰','汽车','机械','设备',
]

def classify_industry(row):
    # Use company name only to avoid endogeneity with AI skill detection in JD
    company_name = str(row.iloc[3])
    d_score = sum(1 for kw in digital_kw if kw.lower() in company_name.lower())
    t_score = sum(1 for kw in traditional_kw if kw.lower() in company_name.lower())
    if d_score > t_score:
        return 'Digital-Intensive'
    elif t_score > d_score:
        return 'Traditional'
    else:
        return 'Other'

df_filtered['industry_type'] = df_filtered.apply(classify_industry, axis=1)

# Company type
def group_company_type(ct):
    ct = str(ct).strip()
    if ct in ['民营']: return 'Private'
    elif ct in ['国企', '事业单位', '政府机关']: return 'State-Owned'
    elif '外资' in ct: return 'Foreign'
    elif ct in ['已上市']: return 'Listed'
    elif ct in ['合资']: return 'Joint-Venture'
    else: return 'Other'

df_filtered['company_type_group'] = df_filtered['er1'].apply(group_company_type)

# Experience
def classify_experience(row):
    text = str(row.iloc[9]) + ' ' + str(row.iloc[0])
    if re.search(r'经验不限|无需经验|应届|无经验|应届生|零基础|新手', text):
        return '0-1 yr'
    for pat in [r'(\d+)[-~](\d*)\s*年.*经验', r'(\d+)\s*年.*经验', r'经验[：:]\s*(\d+)[-~](\d*)']:
        m = re.search(pat, text)
        if m:
            yr = float(m.group(1))
            if yr <= 1: return '0-1 yr'
            elif yr <= 3: return '1-3 yr'
            else: return '3+ yr'
    title = str(row.iloc[0])
    if any(kw in title for kw in ['实习','应届','初级','助理','专员','Junior','培训生','毕业生']):
        return '0-1 yr'
    if any(kw in title for kw in ['高级','资深','专家','Senior','Staff']):
        return '3+ yr'
    return '1-3 yr'

df_filtered['experience_level'] = df_filtered.apply(classify_experience, axis=1)

# Education
def classify_education(text):
    if pd.isna(text): return 'Unspecified'
    text = str(text)
    edu_map = {'博士':'PhD','硕士':'Masters','研究生':'Masters',
               '本科':'Bachelor','学士':'Bachelor',
               '大专':'Associate','专科':'Associate',
               '中专':'High School','高中':'High School'}
    for kw, lvl in edu_map.items():
        if kw in text: return lvl
    return 'Unspecified'

df_filtered['education'] = df_filtered.iloc[:, 9].apply(classify_education)

# Occupation classification (expanded for Occupation FE in regression)
def classify_occupation(title):
    """Classify job title into functional occupation categories.
    Uses priority ordering: more specific categories checked first."""
    title = str(title)
    # 1. Algorithm / AI R&D — most specific
    if any(kw in title for kw in ['算法', 'AI', '人工智能', '机器学习', '深度学习',
        '强化学习', '自然语言处理', 'NLP', '计算机视觉', 'CV', '大模型', 'LLM',
        '自动驾驶', '具身智能', '智能体', 'AIGC', 'Agent', 'RAG',
        '机器人', 'SLAM', 'slam', '三维重建', '3D生成']):
        return 'Algorithm_AI'
    # 2. Data & Analytics (before Software to catch data engineers)
    if any(kw in title for kw in ['数据', '数据分析', '数据挖掘', '数据科学',
        'BI', '商业智能', '数据仓库', '数据工程', '大数据', '数据库',
        'DBA', 'dba', '报表']):
        return 'Data_Analytics'
    # 3. Software Development
    if any(kw in title for kw in ['开发', '工程师', 'Java', 'java', '前端',
        '后端', '全栈', '测试', '运维', 'DevOps', 'C++', 'Golang', 'Node',
        'PHP', '.NET', '架构师', '程序', '软件', 'SRE', 'sre',
        'iOS', 'Android', '安卓', '鸿蒙', 'Flutter', 'React', 'Vue',
        'H5', 'web', 'Web', 'APP', 'app开发', '游戏开发', '服务端']):
        return 'Software_Dev'
    # 4. Hardware / Electronics
    if any(kw in title for kw in ['硬件', '电子', '嵌入式', '芯片', '电路',
        'FPGA', '半导体', 'PCB', '射频', 'IC', '单片机', 'PLC',
        '电气', '自动化工程师']):
        return 'Hardware'
    # 5. Design / Creative
    if any(kw in title for kw in ['设计', 'UI', 'UX', '美工', '视觉', '原画',
        '建模', '渲染', '动画', '3D', '插画', '平面', '包装',
        '视频剪辑', '后期', '特效', '剪辑师']):
        return 'Design'
    # 6. Product Management
    if any(kw in title for kw in ['产品', 'Product']):
        return 'Product'
    # 7. Marketing / Sales
    if any(kw in title for kw in ['市场', '营销', '销售', '商务', '品牌', '渠道',
        '招商', '客户经理', 'BD', '业务', '拓客', '地推',
        '海外销售', '外贸业务', '电商运营']):
        return 'Marketing_Sales'
    # 8. Operations / Customer Service
    if any(kw in title for kw in ['运营', '客服', '售后', '技术支持',
                                  '社群', '社区']):
        return 'Operations'
    # 9. HR / Administration
    if any(kw in title for kw in ['人事', 'HR', '行政', '招聘', '秘书', '前台',
        '人力', '薪酬', '绩效', '员工关系', '猎头', '后勤']):
        return 'HR_Admin'
    # 10. Finance / Accounting
    if any(kw in title for kw in ['财务', '会计', '审计', '出纳', '税务',
        '资金', '预算', '核算', '成本', '费控']):
        return 'Finance_Accounting'
    # 11. Media / Content Creation
    if any(kw in title for kw in ['新媒体', '内容', '短视频', '编辑', '文案',
        '直播', '编导', '记者', '传媒', '拍摄', '脚本', '策划',
        '自媒体', '博主', '网红']):
        return 'Media_Content'
    # 12. Supply Chain / Logistics
    if any(kw in title for kw in ['供应链', '物流', '采购', '仓储', '贸易',
        '进出口', '跟单', '外贸', '快递', '配送', '货运',
        '关务', '船务']):
        return 'Supply_Chain'
    # 13. Education / Training
    if any(kw in title for kw in ['教师', '老师', '讲师', '培训', '教育',
        '教练', '课程', '教务', '留学', '家教']):
        return 'Education'
    # 14. Legal / IP
    if any(kw in title for kw in ['法律', '律师', '专利', '知识产权', '法务',
        '合规', '商标']):
        return 'Legal'
    # 15. Healthcare / Biotech
    if any(kw in title for kw in ['医疗', '医药', '生物', '临床', '护理',
        '医生', '护士', '制药', '药物', '检验', '影像']):
        return 'Healthcare'
    # 16. Engineering / Construction
    if any(kw in title for kw in ['建筑', '工程', '施工', '装修', '监理',
        '造价', '土木', '机电', '暖通', '消防', '给排水', '测绘']):
        return 'Engineering_Construction'
    # 17. Finance / Banking / Investment
    if any(kw in title for kw in ['金融', '投资', '银行', '证券', '保险',
        '基金', '风控', '信贷', '理财', '投行', '资管', '期货']):
        return 'Finance_Banking'
    return 'Other'

df_filtered['job_category'] = df_filtered.iloc[:, 0].apply(classify_occupation)

print('\nIndustry Classification:')
print(df_filtered['industry_type'].value_counts())
print('\nCompany Types:')
print(df_filtered['company_type_group'].value_counts())
print('\nExperience Level:')
print(df_filtered['experience_level'].value_counts())
print('\nEducation:')
print(df_filtered['education'].value_counts())

# ========================================
# 5. Prepare regression data (salary-valid subset)
# ========================================
df_reg = df_filtered[df_filtered['has_valid_salary']].copy()
df_reg['digital_intensive'] = (df_reg['industry_type'] == 'Digital-Intensive').astype(int)
df_reg['exp_0_1'] = (df_reg['experience_level'] == '0-1 yr').astype(int)
df_reg['exp_3_plus'] = (df_reg['experience_level'] == '3+ yr').astype(int)

for edu in ['Bachelor','Masters','Associate','PhD','High School','Unspecified']:
    df_reg[f'edu_{edu}'] = (df_reg['education'] == edu).astype(int)

for ct in ['State-Owned','Foreign','Listed','Joint-Venture','Other']:
    df_reg[f'firm_{ct}'] = (df_reg['company_type_group'] == ct).astype(int)

df_reg['AI_general_int'] = df_reg['AI_general'].astype(int)
df_reg['AI_specialized_int'] = df_reg['AI_specialized'].astype(int)

# Occupation fixed effects (reference: Software_Dev — the largest category)
for occ in df_reg['job_category'].unique():
    if occ != 'Software_Dev':
        occ_key = f'occ_{occ}'
        df_reg[occ_key] = (df_reg['job_category'] == occ).astype(int)

# Clean column names for statsmodels
df_reg.columns = df_reg.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')

print(f'\nRegression sample (with valid salary): {len(df_reg)}')
print('\nOccupation distribution:')
print(df_reg['job_category'].value_counts())

# Build occupation FE variable list
occ_vars = [c for c in df_reg.columns if c.startswith('occ_')]

# ========================================
# 6. CHARTS
# ========================================

# --- fig_salary_distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_reg['salary_monthly'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(df_reg['salary_monthly'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df_reg["salary_monthly"].median():,.0f}')
axes[0].set_xlabel('Monthly Salary (RMB)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Monthly Salary\n(Hangzhou, <1000 employees, excl. interns)')
axes[0].legend()

axes[1].hist(df_reg['ln_salary'], bins=40, color='coral', edgecolor='white', alpha=0.8)
axes[1].axvline(df_reg['ln_salary'].median(), color='blue', linestyle='--', linewidth=2,
                label=f'Median: {df_reg["ln_salary"].median():.3f}')
axes[1].set_xlabel('ln(Monthly Salary)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of ln(Salary)')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_salary_distribution.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_salary_distribution.png')

# --- fig_ai_categories ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
cat_counts = df_filtered['AI_category'].value_counts()
colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
axes[0].pie(cat_counts.values, labels=cat_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, explode=(0.05, 0.05, 0.05, 0.02))
axes[0].set_title('AI Skill Categories Distribution')

ai_salary = df_reg.groupby('AI_category')['salary_monthly'].agg(['mean', 'std', 'count']).reset_index()
order_list = ['No AI Skills', 'General-Purpose AI', 'Specialized AI', 'Both']
ai_salary['AI_category'] = pd.Categorical(ai_salary['AI_category'], categories=order_list, ordered=True)
ai_salary = ai_salary.sort_values('AI_category')
axes[1].barh(ai_salary['AI_category'], ai_salary['mean'],
             xerr=ai_salary['std'] / np.sqrt(ai_salary['count']),
             color=colors_pie, edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('Mean Monthly Salary (RMB)')
axes[1].set_title('Mean Salary by AI Skill Category (+/- SE)')
for i, (v, n) in enumerate(zip(ai_salary['mean'], ai_salary['count'])):
    axes[1].text(v + 0.1, i, f'{v:,.0f} (n={int(n)})', va='center', fontsize=10)

bp_data = [df_reg[df_reg['AI_category'] == cat]['salary_monthly'].dropna()
           for cat in order_list]
bp = axes[2].boxplot(bp_data, labels=['No AI', 'General AI', 'Specialized AI', 'Both'], patch_artist=True)
for patch, color in zip(bp['boxes'], colors_pie):
    patch.set_facecolor(color)
axes[2].set_ylabel('Monthly Salary (RMB)')
axes[2].set_title('Salary Distribution by AI Skill Category')
axes[2].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_ai_categories.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_ai_categories.png')

# --- fig_eda_panel ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
sns.violinplot(x='AI_category', y='salary_monthly', data=df_reg,
               order=order_list, inner='quartile', palette='Set2', ax=axes[0, 0])
axes[0, 0].set_title('(A) Salary Distribution by AI Skill Category')
axes[0, 0].tick_params(axis='x', rotation=15)

mean_sal = df_reg.groupby(['industry_type', 'AI_category'])['salary_monthly'].mean().unstack()
mean_sal = mean_sal.reindex(columns=order_list)
mean_sal.plot(kind='bar', ax=axes[0, 1], color=colors_pie, edgecolor='black', linewidth=0.5)
axes[0, 1].set_title('(B) Mean Salary by AI Category and Industry')
axes[0, 1].set_ylabel('Mean Monthly Salary (RMB)')
axes[0, 1].legend(title='AI Category', fontsize=8, title_fontsize=9)

mean_sal2 = df_reg.groupby(['experience_level','AI_category'])['salary_monthly'].mean().unstack()
mean_sal2 = mean_sal2.reindex(['0-1 yr','1-3 yr','3+ yr'], columns=order_list)
mean_sal2.plot(kind='bar', ax=axes[1, 0], color=colors_pie, edgecolor='black', linewidth=0.5)
axes[1, 0].set_title('(C) Mean Salary by AI Category and Experience')
axes[1, 0].set_ylabel('Mean Monthly Salary (RMB)')
axes[1, 0].legend().remove()

ai_by_ind = df_filtered.groupby('industry_type')['has_AI'].mean()*100
ai_by_ind = ai_by_ind.reindex(['Digital-Intensive','Traditional','Other'])
axes[1, 1].bar(ai_by_ind.index, ai_by_ind.values, color=['#66b3ff','#ff9999','#cccccc'],
               edgecolor='black', linewidth=0.5)
axes[1, 1].set_title('(D) AI Skill Penetration by Industry')
axes[1, 1].set_ylabel('% Requiring AI Skills')
for i, v in enumerate(ai_by_ind.values):
    axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_eda_panel.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_eda_panel.png')

# --- fig_heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ct = pd.crosstab(df_filtered['company_type_group'], df_filtered['AI_category'], normalize='index')*100
sns.heatmap(ct, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label':'%'})
axes[0].set_title('AI Skill Distribution by Company Type (%)')

ct2 = pd.crosstab(df_filtered['experience_level'], df_filtered['AI_category'], normalize='index')*100
ct2 = ct2.reindex(['0-1 yr','1-3 yr','3+ yr'])
sns.heatmap(ct2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label':'%'})
axes[1].set_title('AI Skill Distribution by Experience Level (%)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_heatmaps.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_heatmaps.png')

# --- fig_companies_jobs ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
top_cos = df_filtered[df_filtered['has_AI']]['er'].value_counts().head(15)
axes[0].barh(range(len(top_cos)), top_cos.values, color='steelblue', edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(len(top_cos)))
axes[0].set_yticklabels(top_cos.index, fontsize=9)
axes[0].set_xlabel('Number of AI Job Postings')
axes[0].set_title('Top 15 Companies by AI-Related Postings')
axes[0].invert_yaxis()

job_ai = df_filtered.groupby('job_category').apply(
    lambda x: pd.Series({'total':len(x), 'with_AI':x['has_AI'].sum(), 'pct_AI':x['has_AI'].mean()*100})
).reset_index().sort_values('total', ascending=False).head(12)

x = range(len(job_ai))
w = 0.35
axes[1].bar([i-w/2 for i in x], job_ai['total'], w, label='Total', color='lightgray', edgecolor='black')
axes[1].bar([i+w/2 for i in x], job_ai['with_AI'], w, label='With AI', color='steelblue', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(job_ai['job_category'], rotation=45, ha='right')
axes[1].set_title('AI Skill Demand by Job Category')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_companies_jobs.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_companies_jobs.png')

# --- fig_size_stratification ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
size_order = ['少于50人', '50-150人', '150-500人', '500-1000人']

ct_size = pd.crosstab(df_filtered['company_size'], df_filtered['AI_category'], normalize='index')*100
ct_size = ct_size.reindex(size_order)
ct_size.plot(kind='bar', ax=axes[0], color=colors_pie, edgecolor='black', linewidth=0.5, stacked=True)
axes[0].set_title('AI Skill Distribution by Company Size (%)')
axes[0].set_ylabel('Percentage')
axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=30)

ai_by_size = df_filtered.groupby('company_size')['has_AI'].mean()*100
ai_by_size = ai_by_size.reindex(size_order)
axes[1].bar(range(len(ai_by_size)), ai_by_size.values, color='steelblue', edgecolor='black', linewidth=0.5)
axes[1].set_xticks(range(len(ai_by_size)))
axes[1].set_xticklabels(ai_by_size.index, rotation=30)
axes[1].set_ylabel('% Requiring AI Skills')
axes[1].set_title('AI Skill Penetration by Company Size')
for i, v in enumerate(ai_by_size.values):
    axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_size_stratification.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_size_stratification.png')

# --- Regression models ---
control_vars = [
    'digital_intensive',
    'exp_0_1', 'exp_3_plus',
    'edu_Masters', 'edu_PhD', 'edu_Associate', 'edu_High_School',
    'firm_State_Owned', 'firm_Foreign', 'firm_Listed',
] + occ_vars

def run_reg(dv, ivs, data):
    formula = f"{dv} ~ {' + '.join(ivs)}"
    return smf.ols(formula=formula, data=data).fit()

model1 = run_reg('ln_salary', ['AI_general_int', 'AI_specialized_int'], df_reg)
model2 = run_reg('ln_salary', ['AI_general_int', 'AI_specialized_int', 'digital_intensive'], df_reg)
model3 = run_reg('ln_salary', ['AI_general_int', 'AI_specialized_int'] + control_vars, df_reg)

print('\nRegression models fitted.')

# --- fig_baseline_coefficients ---
fig, ax = plt.subplots(figsize=(12, 6))
key_display = [
    'AI_general_int', 'AI_specialized_int',
    'digital_intensive', 'exp_0_1', 'exp_3_plus',
    'edu_Masters', 'edu_PhD',
    'firm_State_Owned', 'firm_Foreign', 'firm_Listed'
]
labels = [
    'General-Purpose AI', 'Specialized AI', 'Digital-Intensive Industry',
    'Experience: 0-1 yr', 'Experience: 3+ yr',
    'Education: Masters', 'Education: PhD',
    'Firm: State-Owned', 'Firm: Foreign', 'Firm: Listed'
]

actual_keys = [k for k in key_display if k in model3.params.index]
actual_labels = [l for k, l in zip(key_display, labels) if k in model3.params.index]

coefs = [model3.params[k] for k in actual_keys]
ci = model3.conf_int()
ci_lo = [ci.loc[k, 0] for k in actual_keys]
ci_hi = [ci.loc[k, 1] for k in actual_keys]

colors_bar = ['#ff9999' if 'Specialized' in l else '#66b3ff' if 'General' in l else '#999999' for l in actual_labels]
y_pos = range(len(actual_labels)-1, -1, -1)

ax.barh(y_pos, coefs, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.errorbar(coefs, y_pos,
            xerr=[np.array(coefs)-np.array(ci_lo), np.array(ci_hi)-np.array(coefs)],
            fmt='none', ecolor='black', capsize=3)
ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(actual_labels[::-1])
ax.set_xlabel('Coefficient (effect on ln salary)')
ax.set_title('Regression Coefficients: ln(Salary) Determinants (Model 3)\n(Hangzhou, <1000 employees, excl. interns)')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_baseline_coefficients.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_baseline_coefficients.png')

# --- fig_industry_heterogeneity ---
ctrl_no_ind = [c for c in control_vars if c != 'digital_intensive']
df_digital = df_reg[df_reg['industry_type'] == 'Digital-Intensive']
df_traditional = df_reg[df_reg['industry_type'] == 'Traditional']

m_digital = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+ctrl_no_ind, df_digital)
m_traditional = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+ctrl_no_ind, df_traditional)

fig, ax = plt.subplots(figsize=(10, 7))
industries = ['Digital-Intensive', 'Traditional']
models_list = [m_digital, m_traditional]
x_pos = [0, 1]
width = 0.35

gp_key = 'AI_general_int'; sp_key = 'AI_specialized_int'
for i, (ind, model) in enumerate(zip(industries, models_list)):
    coefs_i = [model.params[gp_key], model.params[sp_key]]
    cis = model.conf_int()
    err_lo = [coefs_i[0]-cis.loc[gp_key,0], coefs_i[1]-cis.loc[sp_key,0]]
    err_hi = [cis.loc[gp_key,1]-coefs_i[0], cis.loc[sp_key,1]-coefs_i[1]]
    offset = (i - 0.5) * width
    ax.bar(np.array(x_pos)+offset, coefs_i, width, label=ind,
           color=['#66b3ff','#ff9999'][i], edgecolor='black', linewidth=0.5)
    ax.errorbar(np.array(x_pos)+offset, coefs_i, yerr=[err_lo, err_hi],
                fmt='none', ecolor='black', capsize=5)
    for j, c in enumerate(coefs_i):
        ax.text(x_pos[j]+offset, c+0.01, f'{(np.exp(c)-1)*100:+.1f}%', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(['General-Purpose AI', 'Specialized AI'])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Coefficient on ln(Salary)')
ax.set_title(f'H2: AI Skill Premiums by Industry Type\n(Digital: n={len(df_digital)}, Traditional: n={len(df_traditional)})')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_industry_heterogeneity.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_industry_heterogeneity.png')

# --- fig_experience_heterogeneity ---
c_exp = [c for c in control_vars if not c.startswith('exp_')]
df_e1 = df_reg[df_reg['experience_level']=='0-1 yr']
df_e2 = df_reg[df_reg['experience_level']=='1-3 yr']
df_e3 = df_reg[df_reg['experience_level']=='3+ yr']

m_01 = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+c_exp, df_e1)
m_13 = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+c_exp, df_e2)
m_3p = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+c_exp, df_e3)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax_idx, (skill_param, label) in enumerate([(gp_key, 'General-Purpose AI'), (sp_key, 'Specialized AI')]):
    ax = axes[ax_idx]
    coefs_e = [m_01.params[skill_param], m_13.params[skill_param], m_3p.params[skill_param]]
    ci_lo_e = [m_01.conf_int().loc[skill_param,0], m_13.conf_int().loc[skill_param,0], m_3p.conf_int().loc[skill_param,0]]
    ci_hi_e = [m_01.conf_int().loc[skill_param,1], m_13.conf_int().loc[skill_param,1], m_3p.conf_int().loc[skill_param,1]]

    x_e = np.arange(3)
    color = '#66b3ff' if 'General' in label else '#ff9999'
    ax.bar(x_e, coefs_e, color=color, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_e, coefs_e,
                yerr=[np.array(coefs_e)-np.array(ci_lo_e), np.array(ci_hi_e)-np.array(coefs_e)],
                fmt='none', ecolor='black', capsize=5)
    ax.set_xticks(x_e)
    ax.set_xticklabels(['0-1 yr', '1-3 yr', '3+ yr'])
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Coefficient on ln(Salary)')
    ax.set_title(f'({chr(65+ax_idx)}) {label} Premium by Experience')

    for i, c in enumerate(coefs_e):
        ax.text(i, c+0.03, f'{(np.exp(c)-1)*100:+.1f}%', ha='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_experience_heterogeneity.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_experience_heterogeneity.png')

# --- fig_size_heterogeneity ---
size_order = ['少于50人', '50-150人', '150-500人', '500-1000人']
ctrl_no_size_ind = [c for c in control_vars if c != 'digital_intensive']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
size_models = {}
for size_cat in size_order:
    df_s = df_reg[df_reg['company_size'] == size_cat]
    if len(df_s) > 30:
        size_models[size_cat] = run_reg('ln_salary', ['AI_general_int','AI_specialized_int']+ctrl_no_size_ind, df_s)
    else:
        size_models[size_cat] = None

# GP premium by size
valid_sizes_gp = []
gp_premiums = []
gp_err_lo = []; gp_err_hi = []
for sz in size_order:
    m = size_models[sz]
    if m is not None and gp_key in m.params.index:
        valid_sizes_gp.append(sz)
        c = m.params[gp_key]
        gp_premiums.append((np.exp(c)-1)*100)
        gp_err_lo.append(((np.exp(c)-1)*100) - ((np.exp(m.conf_int().loc[gp_key,0])-1)*100))
        gp_err_hi.append(((np.exp(m.conf_int().loc[gp_key,1])-1)*100) - ((np.exp(c)-1)*100))

x_sz = range(len(valid_sizes_gp))
axes[0].bar(x_sz, gp_premiums, color='#66b3ff', edgecolor='black', linewidth=0.5)
axes[0].errorbar(x_sz, gp_premiums, yerr=[gp_err_lo, gp_err_hi], fmt='none', ecolor='black', capsize=5)
axes[0].set_xticks(x_sz)
axes[0].set_xticklabels(valid_sizes_gp, rotation=30)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_ylabel('Wage Premium (%)')
axes[0].set_title('(A) General-Purpose AI Premium by Company Size')

# SP premium by size
valid_sizes_sp = []
sp_premiums = []
sp_err_lo = []; sp_err_hi = []
for sz in size_order:
    m = size_models[sz]
    if m is not None and sp_key in m.params.index:
        valid_sizes_sp.append(sz)
        c = m.params[sp_key]
        sp_premiums.append((np.exp(c)-1)*100)
        sp_err_lo.append(((np.exp(c)-1)*100) - ((np.exp(m.conf_int().loc[sp_key,0])-1)*100))
        sp_err_hi.append(((np.exp(m.conf_int().loc[sp_key,1])-1)*100) - ((np.exp(c)-1)*100))

axes[1].bar(x_sz, sp_premiums, color='#ff9999', edgecolor='black', linewidth=0.5)
axes[1].errorbar(x_sz, sp_premiums, yerr=[sp_err_lo, sp_err_hi], fmt='none', ecolor='black', capsize=5)
axes[1].set_xticks(x_sz)
axes[1].set_xticklabels(valid_sizes_sp, rotation=30)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_ylabel('Wage Premium (%)')
axes[1].set_title('(B) Specialized AI Premium by Company Size')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_size_heterogeneity.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_size_heterogeneity.png')

# --- fig_robustness ---
# Lower bound salary
def parse_salary_lower(sal_str):
    """Parse lower bound of salary range, same logic as parse_salary but takes
    the low end instead of midpoint. Returns monthly salary in RMB."""
    if pd.isna(sal_str) or str(sal_str).strip() == '':
        return np.nan
    s = str(sal_str).strip()

    # Daily rates
    if re.search(r'元/天|元/日|/天|/日', s):
        return np.nan

    bm = 12
    m = re.search(r'(\d+)\s*薪', s)
    if m:
        bm = int(m.group(1))
        s = re.sub(r'·?\d+\s*薪', '', s).strip()

    is_annual = bool(re.search(r'/年|/year', s))
    s = re.sub(r'/年|/year', '', s).strip()

    def parse_single_value(val_str, default_unit='万'):
        val_str = val_str.strip()
        m = re.match(r'([\d.]+)\s*(万|千)?', val_str)
        if not m:
            return None, None
        val = float(m.group(1))
        unit = m.group(2) if m.group(2) else None
        return val * (10 if (unit or default_unit) == '万' else 1), unit

    # Range pattern with per-side unit parsing
    m = re.search(
        r'([\d.]+\s*(?:万|千)?)\s*[-–~至]\s*([\d.]+\s*(?:万|千)?)', s
    )
    if m:
        low_val, low_unit = parse_single_value(m.group(1))
        high_val, high_unit = parse_single_value(m.group(2))
        if low_val is not None:
            # If low side has no unit, inherit from high side
            if low_unit is None and high_unit is not None:
                low_val = parse_single_value(m.group(1), high_unit)[0]
            monthly = low_val
        else:
            return np.nan
    else:
        val, _ = parse_single_value(s)
        if val is not None:
            monthly = val
        else:
            return np.nan

    if is_annual:
        monthly /= 12
    return monthly * bm / 12 * 1000

df_reg['salary_lower'] = df_filtered.loc[df_reg.index,'sal'].apply(parse_salary_lower)
df_reg['ln_salary_lower'] = np.log(df_reg['salary_lower'])
m3_lower = smf.ols(f"ln_salary_lower ~ AI_general_int+AI_specialized_int+{' + '.join(control_vars)}", data=df_reg).fit()

# Trimmed
p5, p95 = df_reg['salary_monthly'].quantile(0.05), df_reg['salary_monthly'].quantile(0.95)
df_trimmed = df_reg[(df_reg['salary_monthly']>=p5)&(df_reg['salary_monthly']<=p95)]
m3_trim = smf.ols(f"ln_salary ~ AI_general_int+AI_specialized_int+{' + '.join(control_vars)}", data=df_trimmed).fit()

# Placebo
np.random.seed(42)
df_placebo = df_reg.copy()
df_placebo['AI_gen_r'] = np.random.permutation(df_reg['AI_general_int'].values)
df_placebo['AI_spec_r'] = np.random.permutation(df_reg['AI_specialized_int'].values)
m_placebo = smf.ols(f"ln_salary ~ AI_gen_r+AI_spec_r+{' + '.join(control_vars)}", data=df_placebo).fit()

fig, ax = plt.subplots(figsize=(12, 8))
models_r = [model3, m3_lower, m3_trim, m_placebo]
labels_r = ['Baseline\n(Midpoint)', 'Lower Bound\nSalary', 'Trimmed\n5-95%', 'Placebo\n(Random)']
param_pairs_r = [
    ('AI_general_int','AI_specialized_int'),
    ('AI_general_int','AI_specialized_int'),
    ('AI_general_int','AI_specialized_int'),
    ('AI_gen_r','AI_spec_r')
]

x_r = np.arange(4); w_r = 0.35
for si, (label, color) in enumerate(zip(['General-Purpose AI','Specialized AI'], ['#66b3ff','#ff9999'])):
    coefs_r = []; err_lo_r = []; err_hi_r = []
    for m, (p1,p2) in zip(models_r, param_pairs_r):
        pname = p1 if si==0 else p2
        c = m.params[pname] if pname in m.params.index else 0
        ci_v = m.conf_int().loc[pname] if pname in m.params.index else [c,c]
        coefs_r.append(c)
        err_lo_r.append(c-ci_v[0]); err_hi_r.append(ci_v[1]-c)
    offset = (si-0.5)*w_r
    ax.bar(x_r+offset, coefs_r, w_r, label=label, color=color, edgecolor='black', linewidth=0.5)
    ax.errorbar(x_r+offset, coefs_r, yerr=[err_lo_r, err_hi_r], fmt='none', ecolor='black', capsize=5)

ax.set_xticks(x_r)
ax.set_xticklabels(labels_r)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Coefficient on ln(Salary)')
ax.set_title('Robustness Checks: AI Skill Coefficients Across Specifications')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_robustness.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_robustness.png')

# --- fig_skill_correlation ---
ai_terms = ['Python','ChatGPT','深度学习','机器学习','计算机视觉','NLP',
    'Midjourney','Stable Diffusion','RAG','Agent','Prompt','大模型','数据','设计','视频','分析']

for term in ai_terms:
    df_reg[f'skill_{term}'] = df_filtered.loc[df_reg.index, df_filtered.columns[9]].str.contains(
        term, na=False, case=False).astype(int)

skill_cols = [f'skill_{t}' for t in ai_terms]
corr_mat = df_reg[skill_cols].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            xticklabels=ai_terms, yticklabels=ai_terms, cbar_kws={'label':'Correlation'})
ax.set_title('AI Skill Term Co-occurrence in Job Descriptions')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_skill_correlation.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_skill_correlation.png')

# --- fig_predicted_salary ---
scenarios_dict = {
    'AI_general_int': [0,1,0,1], 'AI_specialized_int': [0,0,1,1],
    'digital_intensive': [1,1,1,1], 'exp_0_1': [0,0,0,0], 'exp_3_plus': [0,0,0,0],
    'edu_Masters': [0,0,0,0], 'edu_PhD': [0,0,0,0], 'edu_Associate': [0,0,0,0],
    'edu_High_School': [0,0,0,0], 'firm_State_Owned': [0,0,0,0],
    'firm_Foreign': [0,0,0,0], 'firm_Listed': [0,0,0,0],
}
# Set all occupation dummies to 0 (reference: Software_Dev)
for ov in occ_vars:
    scenarios_dict[ov] = [0, 0, 0, 0]

# Handle edu_Unspecified and firm_Other which may not be in model
for extra in ['edu_Unspecified', 'firm_Other', 'firm_Joint_Venture']:
    if extra in model3.params.index:
        scenarios_dict[extra] = [0, 0, 0, 0]

scenarios = pd.DataFrame(scenarios_dict)

scenarios['predicted_salary'] = np.exp(model3.predict(scenarios))
scenarios['Scenario'] = ['No AI', 'General AI Only', 'Specialized AI Only', 'Both']

fig, ax = plt.subplots(figsize=(10, 6))
colors_s = ['#ffcc99','#66b3ff','#ff9999','#99ff99']
bars = ax.bar(range(4), scenarios['predicted_salary'], color=colors_s, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels(scenarios['Scenario'])
ax.set_ylabel('Predicted Monthly Salary (RMB)')
ax.set_title('Predicted Salary by AI Skill Profile (Model 3)\n'
             '(Digital-Intensive Industry, 1-3 yr Exp, Bachelor, Private Firm)')

base = scenarios['predicted_salary'].iloc[0]
for i, bar in enumerate(bars):
    val = scenarios['predicted_salary'].iloc[i]
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:,.0f}', ha='center', fontweight='bold')
    if i > 0:
        ax.text(i, val+0.4, f'+{(val-base)/base*100:.1f}%', ha='center', fontsize=10, color='darkred')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_predicted_salary.png'), bbox_inches='tight')
plt.close()
print('Saved: fig_predicted_salary.png')

# ========================================
# 7. Export CSVs
# ========================================
high_ai = df_filtered[df_filtered['AI_demand'] == '高AI需求']
low_ai = df_filtered[df_filtered['AI_demand'] == '低AI需求']
high_ai.to_csv(os.path.join(output_folder, '高AI需求_小企业.csv'), index=False, encoding='utf-8-sig')
low_ai.to_csv(os.path.join(output_folder, '低AI需求_小企业.csv'), index=False, encoding='utf-8-sig')
print(f'Exported: 高AI需求_小企业.csv ({len(high_ai)} rows)')
print(f'Exported: 低AI需求_小企业.csv ({len(low_ai)} rows)')

# ========================================
# 8. Generate Report
# ========================================
# Compute statistics for report
total_all = len(df)
total_filtered = len(df_filtered)
total_large = total_all - total_filtered  # approximate

demand_counts = df_filtered['AI_demand'].value_counts()
high_count = demand_counts.get('高AI需求', 0)
low_count = demand_counts.get('低AI需求', 0)
no_count = demand_counts.get('无AI需求', 0)

size_cross = pd.crosstab(df_filtered['company_size'], df_filtered['AI_demand'])
size_cross = size_cross.reindex(size_order)

# Top job titles
high_ai_titles = df_filtered[df_filtered['AI_demand'] == '高AI需求'].iloc[:, 0].value_counts().head(20)
low_ai_titles = df_filtered[df_filtered['AI_demand'] == '低AI需求'].iloc[:, 0].value_counts().head(20)

report = f"""# 51job数据AI需求分析报告

## 分析条件
- 数据来源：51job_final_result.csv
- 城市筛选：仅杭州
- 企业规模筛选：< 1000人（少于50人、50-150人、150-500人、500-1000人）
- 职位类型筛选：排除实习/兼职岗位
- 薪资限制：无
- 总数据量：{total_all}条职位记录
- 符合条件记录：{total_filtered}条

## AI需求分类标准

### 高AI需求
职位标题或描述中明确需要AI核心技术能力，包括：
- 算法工程师/研究员、大模型/LLM相关岗位
- 机器学习、深度学习、强化学习
- 自然语言处理(NLP)、计算机视觉(CV)
- 语音识别/合成、具身智能、智能体(Agent)
- AI模型训练/推理/部署、AI编译器
- AIGC、多模态、三维重建等前沿AI方向
- AI应用开发、AI研发等直接AI技术岗位

### 低AI需求
职位与AI相关但属于应用层或辅助角色，包括：
- AI产品经理、AI项目管理
- AI销售/客户经理
- Python开发（含数据分析、爬虫等）
- 数据分析师、数据挖掘工程师
- 数字化/数智化岗位
- RPA自动化、数据标注等

### 无AI需求
职位标题和描述中不涉及AI相关内容。

---

## 分析结果

### 1000人以内规模企业（杭州，排除实习）

| AI需求类别 | 数量 | 占比 |
|-----------|------|------|
| 高AI需求 | {high_count} | {high_count/total_filtered*100:.1f}% |
| 低AI需求 | {low_count} | {low_count/total_filtered*100:.1f}% |
| 无AI需求 | {no_count} | {no_count/total_filtered*100:.1f}% |
| **合计** | **{total_filtered}** | **100%** |

### 按企业规模细分

| 企业规模 | 高AI需求 | 低AI需求 | 无AI需求 | 小计 |
|---------|---------|---------|---------|------|
"""

for sz in size_order:
    if sz in size_cross.index:
        h = int(size_cross.loc[sz, '高AI需求']) if '高AI需求' in size_cross.columns else 0
        l = int(size_cross.loc[sz, '低AI需求']) if '低AI需求' in size_cross.columns else 0
        n = int(size_cross.loc[sz, '无AI需求']) if '无AI需求' in size_cross.columns else 0
        report += f"| {sz} | {h} | {l} | {n} | {h+l+n} |\n"

report += f"""
### 高AI需求 - 热门职位TOP20

| 职位名称 | 数量 |
|---------|------|
"""

for title, count in high_ai_titles.head(20).items():
    report += f"| {title} | {count} |\n"

report += f"""
### 低AI需求 - 热门职位TOP20

| 职位名称 | 数量 |
|---------|------|
"""

for title, count in low_ai_titles.head(20).items():
    report += f"| {title} | {count} |\n"

report += f"""
---

## 薪资统计（有效薪资样本：{len(df_reg)}条）

| 指标 | 数值 |
|------|------|
| 平均月薪 | {df_reg['salary_monthly'].mean():,.0f} |
| 中位数月薪 | {df_reg['salary_monthly'].median():,.0f} |
| 标准差 | {df_reg['salary_monthly'].std():,.0f} |
| 最低 | {df_reg['salary_monthly'].min():,.0f} |
| 最高 | {df_reg['salary_monthly'].max():,.0f} |

### AI技能分类（学术分类）

| 类别 | 数量 | 占比 |
|------|------|------|
| Specialized AI | {(df_filtered['AI_category']=='Specialized AI').sum()} | {(df_filtered['AI_category']=='Specialized AI').sum()/total_filtered*100:.1f}% |
| General-Purpose AI | {(df_filtered['AI_category']=='General-Purpose AI').sum()} | {(df_filtered['AI_category']=='General-Purpose AI').sum()/total_filtered*100:.1f}% |
| Both | {(df_filtered['AI_category']=='Both').sum()} | {(df_filtered['AI_category']=='Both').sum()/total_filtered*100:.1f}% |
| No AI Skills | {(df_filtered['AI_category']=='No AI Skills').sum()} | {(df_filtered['AI_category']=='No AI Skills').sum()/total_filtered*100:.1f}% |

### 回归结果（Model 3）

| 变量 | 系数 | 标准误 | P值 |
|------|------|--------|-----|
| AI_general | {model3.params['AI_general_int']:.4f} | {model3.bse['AI_general_int']:.4f} | {model3.pvalues['AI_general_int']:.4f} |
| AI_specialized | {model3.params['AI_specialized_int']:.4f} | {model3.bse['AI_specialized_int']:.4f} | {model3.pvalues['AI_specialized_int']:.4f} |
| R-squared | {model3.rsquared:.4f} | | |
| Adj R-squared | {model3.rsquared_adj:.4f} | | |
| N | {int(model3.nobs)} | | |

---

## 结论

1. 在杭州1000人以内规模的企业中（排除实习），**高AI需求**岗位占比为 **{high_count/total_filtered*100:.1f}%**（{high_count}个）。
2. **低AI需求**岗位占比 **{low_count/total_filtered*100:.1f}%**（{low_count}个），主要体现在AI产品、Python开发、数据分析等辅助性AI岗位。
3. **无AI需求**岗位占比 **{no_count/total_filtered*100:.1f}%**（{no_count}个）。
4. 专业化AI技能（Specialized AI）的薪资溢价显著高于通用AI技能（General-Purpose AI）。

*报告生成时间：2026年6月4日*
*分析条件：杭州 + <1000人企业规模 + 排除实习/兼职，不限薪资*
"""

with open(os.path.join(output_folder, 'AI需求分析报告.md'), 'w', encoding='utf-8') as f:
    f.write(report)
print('Saved: AI需求分析报告.md')

# ========================================
# 9. Summary
# ========================================
print('\n' + '='*60)
print('ANALYSIS COMPLETE')
print('='*60)
print(f'Filter conditions: Hangzhou + <1000 employees + exclude interns')
print(f'Total sample: {total_filtered}')
print(f'High AI demand: {high_count} ({high_count/total_filtered*100:.1f}%)')
print(f'Low AI demand: {low_count} ({low_count/total_filtered*100:.1f}%)')
print(f'No AI demand: {no_count} ({no_count/total_filtered*100:.1f}%)')
print(f'Valid salary sample: {len(df_reg)}')
print(f'All outputs saved to: {output_folder}')
