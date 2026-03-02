import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pickle
import os
import warnings
import datetime
import xgboost as xgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class 麻醉反馈分析系统:
    def __init__(self, root):
        self.root = root
        self.root.title("麻醉科住培反馈效用评价系统（四级分类）- 稳定版")
        self.root.geometry("1200x800")
        
        # 初始化模型和变量
        self.效用模型 = None
        self.向量化器 = None
        self.标准化器 = None
        self.预处理数据 = None
        self.专业词库文件 = "麻醉专业词库.txt"
        self.停用词列表 = self.获取停用词列表()
        self.训练性能结果 = None
        self.训练特征维度 = None
        self.混淆矩阵 = None
        self.快速模式 = False
        
        # 移除BERT相关初始化
        self.使用深度特征 = False
        self.BERT可用 = False
        
        self.初始化专业词库()
        
        # 高质量特征关键词 - 增强版
        self.高质量特征关键词 = {
            "以住培医师行为为中心的": ["行为", "操作", "表现", "动作", "医师", "医生", "学员", "住院医师", "规培", "住培"],
            "提供详细信息的": ["详细", "具体", "明确", "清晰", "描述", "记录", "记载", "说明", "步骤", "流程", "方法", "技术"], 
            "负面反馈": ["不足", "改进", "错误", "问题", "缺点", "欠缺", "需要提高", "不够", "未", "缺乏", "差", "不好", "有待", "提高空间", "欠缺"],
            "评价专业性和沟通能力的": ["专业", "沟通", "交流", "解释", "告知", "说明", "计划", "准备", "能力", "技能", "沟通能力", "交流能力", "专业能力"],
            "有针对性的": ["针对", "特定", "具体", "相关", "对应", "针对性地", "专门", "特别", "特定", "具体问题"],
            "指出可改进的": ["改进", "提高", "加强", "完善", "优化", "建议", "应该", "需要", "要", "可以", "改进方向", "建议", "改进建议"],
            "具体操作行为": ["操作", "执行", "完成", "掌握", "学习", "训练", "练习", "处理", "管理", "评估", "诊断", "治疗"],
            "专业知识相关": ["理论", "知识", "技能", "专业", "临床", "实践", "基础", "经验", "水平", "能力"],
            "麻醉专业术语": ["麻醉", "镇痛", "镇静", "全麻", "局麻", "腰麻", "硬膜外", "插管", "呼吸机", "监护", "苏醒", "复苏"]
        }
        
        # 麻醉专业关键词库
        self.麻醉关键词 = {
            "麻醉操作": ["气管插管", "静脉穿刺", "动脉穿刺", "椎管内麻醉", "神经阻滞", "喉罩", "纤支镜"],
            "药物": ["丙泊酚", "芬太尼", "舒芬太尼", "罗库溴铵", "顺式阿曲库铵", "七氟烷", "异氟烷", "地氟烷"],
            "监测": ["心电图", "血压", "血氧饱和度", "呼气末二氧化碳", "体温", "麻醉深度", "肌松监测"],
            "并发症": ["术中知晓", "术后恶心呕吐", "困难气道", "过敏反应", "低血压", "高血压", "心律失常"],
            "患者管理": ["术前访视", "术后随访", "镇痛管理", "液体管理", "呼吸管理", "循环管理"]
        }
        
        # 创建界面
        self.创建界面()
        
    def 初始化专业词库(self):
        """初始化或加载麻醉专业词库"""
        基础词汇 = [
            "纤支镜", "双腔气管插管", "麻醉诱导", "麻醉维持", "麻醉复苏",
            "硬膜外麻醉", "腰麻", "全麻", "局麻", "镇静镇痛",
            "呼吸机", "血氧饱和度", "心电图", "血压监测", "气道管理",
            "术中知晓", "术后恶心呕吐", "困难气道", "快速顺序诱导",
            "丙泊酚", "芬太尼", "罗库溴铵", "七氟烷", "依托咪酯",
            "喉罩", "气管导管", "中心静脉穿刺", "动脉穿刺", "椎管内麻醉",
            "臂丛阻滞", "术后镇痛", "麻醉深度", "血流动力学", "呼吸循环",
            "麻醉机", "监护仪", "肌松监测", "脑电双频指数", "镇痛泵",
            "术前访视", "术后随访", "麻醉记录", "麻醉计划", "麻醉并发症",
            "困难插管", "喉痉挛", "支气管痉挛", "过敏反应", "休克",
            "心肺复苏", "除颤", "肾上腺素", "阿托品", "麻黄碱",
            "腰硬联合", "连续硬膜外", "骶管麻醉", "臂丛神经阻滞", "颈丛神经阻滞"
        ]
        
        if not os.path.exists(self.专业词库文件):
            with open(self.专业词库文件, 'w', encoding='utf-8') as f:
                for word in 基础词汇:
                    f.write(word + '\n')
        
        # 加载专业词库到jieba
        jieba.load_userdict(self.专业词库文件)
    
    def 添加专业词汇(self, 新词汇列表):
        """向专业词库添加新词汇"""
        try:
            with open(self.专业词库文件, 'a', encoding='utf-8') as f:
                for word in 新词汇列表:
                    f.write(word + '\n')
            # 重新加载词库
            jieba.load_userdict(self.专业词库文件)
            return True
        except Exception as e:
            print(f"添加词汇失败: {e}")
            return False
    
    def 文本预处理(self, 文本):
        """增强的文本预处理"""
        if pd.isna(文本) or 文本 is None:
            return ""
        
        if isinstance(文本, str) and len(文本.strip()) == 0:
            return ""
        
        # 移除特殊字符和多余空格
        文本 = re.sub(r'[^\w\u4e00-\u9fff。，；：！？、\-\+\%\$\#\@\&\*\(\)\[\]\{\}]', ' ', str(文本))
        文本 = re.sub(r'\s+', ' ', 文本).strip()
        
        if len(文本) == 0:
            return ""
        
        try:
            # 使用精确模式分词
            words = jieba.lcut(文本, cut_all=False)
            
            # 过滤短词和停用词
            words = [word for word in words if len(word) > 1 and word not in self.停用词列表]
            
            if len(words) == 0:
                return ""
                
            return ' '.join(words)
        except Exception as e:
            print(f"分词错误: {e}")
            return ""
    
    def 获取停用词列表(self):
        """获取中文停用词列表"""
        停用词 = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', 
            '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', 
            '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', 
            '而', '但', '与', '之', '或', '且', '及', '并', '对于', '关于', 
            '对', '从', '向', '以', '为', '于', '把', '被', '让', '给', '使',
            '可以', '可能', '能够', '应该', '必须', '需要', '要求', '一定',
            '我们', '你们', '他们', '她们', '它们', '什么', '怎么', '为什么',
            '这个', '那个', '这些', '那些', '这样', '那样', '这里', '那里',
            '啊', '哦', '嗯', '呃', '唉', '呀', '嘛', '吧', '呢', '啦', '哇'
        ]
        return 停用词
    
    def 提取文本特征(self, 文本列表, 训练模式=True):
        """从文本中提取TF-IDF特征"""
        有效文本列表 = [文本 for 文本 in 文本列表 if 文本 and len(文本.strip()) > 0]
        
        if len(有效文本列表) == 0:
            return np.array([]).reshape(len(文本列表), 0)
        
        if 训练模式:
            # 训练模式：创建新的向量化器
            self.向量化器 = TfidfVectorizer(
                max_features=500,  # 增加特征数量
                min_df=2,  # 过滤低频词
                max_df=0.8,
                ngram_range=(1, 2),  # 使用1-gram和2-gram
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True
            )
            try:
                特征矩阵 = self.向量化器.fit_transform(有效文本列表)
            except Exception as e:
                print(f"TF-IDF特征提取失败: {e}")
                return np.array([]).reshape(len(文本列表), 0)
        else:
            # 测试模式：使用训练时的向量化器
            if self.向量化器 is None:
                print("警告：向量化器未初始化")
                return np.array([]).reshape(len(文本列表), 0)
            
            try:
                特征矩阵 = self.向量化器.transform(有效文本列表)
            except Exception as e:
                print(f"TF-IDF特征转换失败: {e}")
                if hasattr(self.向量化器, 'vocabulary_'):
                    vocab_size = len(self.向量化器.vocabulary_)
                    特征矩阵 = np.zeros((len(文本列表), vocab_size))
                else:
                    特征矩阵 = np.array([]).reshape(len(文本列表), 0)
                return 特征矩阵
        
        if len(有效文本列表) < len(文本列表):
            完整特征矩阵 = np.zeros((len(文本列表), 特征矩阵.shape[1]))
            有效索引 = [i for i, 文本 in enumerate(文本列表) if 文本 and len(文本.strip()) > 0]
            完整特征矩阵[有效索引] = 特征矩阵.toarray() if hasattr(特征矩阵, 'toarray') else 特征矩阵
            return 完整特征矩阵
        else:
            return 特征矩阵.toarray() if hasattr(特征矩阵, 'toarray') else 特征矩阵
    
    def 增强_具体性矫正性特征(self, 文本列表):
        """增强的具体性和矫正性特征提取 - 修复版本"""
        特征矩阵 = []
        
        for 文本 in 文本列表:
            features = []
            
            if not 文本 or len(文本.strip()) == 0:
                # 固定15个特征（根据你的代码统计）
                特征矩阵.append([0.0] * 15)  # 使用float类型
                continue
            
            # 1. 具体性特征
            具体性关键词 = ["具体", "详细", "明确", "清晰", "描述", "记录", 
                      "步骤", "操作", "方法", "流程", "技术", "细节"]
            具体性得分 = sum(1 for keyword in 具体性关键词 if keyword in 文本)
            features.append(float(具体性得分))  # 转换为float
            features.append(具体性得分 / max(1.0, float(len(文本.split()))))  # 归一化
            
            # 2. 矫正性特征
            矫正性关键词 = ["改进", "提高", "加强", "完善", "优化", "建议",
                      "应该", "需要", "要", "可以", "改进方向", "不足",
                      "问题", "缺点", "欠缺", "提高空间", "有待", "建议改进"]
            矫正性得分 = sum(1 for keyword in 矫正性关键词 if keyword in 文本)
            features.append(float(矫正性得分))
            features.append(矫正性得分 / max(1.0, float(len(文本.split()))))  # 归一化
            
            # 3. 行为动词特征（指示具体操作）
            行为动词 = ["操作", "执行", "完成", "掌握", "学习", "训练",
                   "练习", "处理", "管理", "评估", "诊断", "治疗", "实施"]
            行为动词得分 = sum(1 for verb in 行为动词 if verb in 文本)
            features.append(float(行为动词得分))
            
            # 4. 专业术语密度
            专业词汇列表 = self.获取专业词汇列表()
            专业词计数 = sum(1 for word in 文本.split() if word in 专业词汇列表)
            features.append(float(专业词计数))
            features.append(float(专业词计数) / max(1.0, float(len(文本.split()))))  # 专业术语密度
            
            # 5. 负面词比例
            负面词 = ["不足", "错误", "问题", "缺点", "欠缺", "差", "不好", "缺乏", "未"]
            负面词计数 = sum(1 for word in 负面词 if word in 文本)
            features.append(float(负面词计数) / max(1.0, float(len(文本.split()))))
            
            # 6. 改进建议强度（矫正性/具体性组合）
            if 具体性得分 > 0 and 矫正性得分 > 0:
                features.append(1.0)  # 有具体的改进建议
            else:
                features.append(0.0)
            
            # 7. 句子长度特征
            features.append(float(len(文本)))
            features.append(float(len(文本.split())))
            
            # 8. 词汇多样性
            words = 文本.split()
            if len(words) > 0:
                unique_words = set(words)
                features.append(float(len(unique_words)) / float(len(words)))
            else:
                features.append(0.0)
            
            # 9. 数字特征
            digit_count = sum(1 for char in 文本 if char.isdigit())
            features.append(float(digit_count))
            
            # 10. 标点符号特征
            标点符号 = ["。", "，", "；", "：", "！", "？", "、"]
            punctuation_count = sum(1 for char in 文本 if char in 标点符号)
            features.append(float(punctuation_count))
            
            # 确保有15个特征
            if len(features) != 15:
                print(f"警告: 特征数量不正确: {len(features)}，应该是15")
                # 补足或截断
                if len(features) < 15:
                    features.extend([0.0] * (15 - len(features)))
                else:
                    features = features[:15]
            
            特征矩阵.append(features)
        
        # 转换为numpy数组，确保数据类型一致
        try:
            特征数组 = np.array(特征矩阵, dtype=np.float32)
            return 特征数组
        except Exception as e:
            print(f"转换特征数组时出错: {e}")
            # 如果转换失败，创建全零数组
            return np.zeros((len(文本列表), 15), dtype=np.float32)
    
    def 提取麻醉专业特征(self, 文本列表):
        """提取麻醉专业相关特征"""
        特征矩阵 = []
        
        for 文本 in 文本列表:
            features = []
            
            if not 文本 or len(文本.strip()) == 0:
                特征矩阵.append([0] * 8)  # 8个麻醉专业特征
                continue
            
            # 麻醉操作相关词
            操作词 = ["插管", "穿刺", "麻醉", "镇痛", "镇静", "阻滞", "硬膜外", "腰麻"]
            操作词计数 = sum(1 for word in 操作词 if word in 文本)
            features.append(操作词计数)
            
            # 麻醉药物相关词
            药物词 = ["丙泊酚", "芬太尼", "罗库溴铵", "七氟烷", "依托咪酯", "舒芬太尼", "异氟烷"]
            药物词计数 = sum(1 for word in 药物词 if word in 文本)
            features.append(药物词计数)
            
            # 监测相关词
            监测词 = ["心电图", "血压", "血氧", "二氧化碳", "体温", "麻醉深度", "肌松"]
            监测词计数 = sum(1 for word in 监测词 if word in 文本)
            features.append(监测词计数)
            
            # 并发症相关词
            并发症词 = ["知晓", "恶心", "呕吐", "困难气道", "过敏", "低血压", "高血压", "心律失常"]
            并发症词计数 = sum(1 for word in 并发症词 if word in 文本)
            features.append(并发症词计数)
            
            # 患者管理相关词
            管理词 = ["访视", "随访", "镇痛", "液体", "呼吸", "循环", "管理"]
            管理词计数 = sum(1 for word in 管理词 if word in 文本)
            features.append(管理词计数)
            
            # 专业术语总数
            features.append(操作词计数 + 药物词计数 + 监测词计数 + 并发症词计数 + 管理词计数)
            
            # 专业术语密度
            total_words = len(文本.split())
            features.append((操作词计数 + 药物词计数 + 监测词计数 + 并发症词计数 + 管理词计数) / max(1, total_words))
            
            # 是否有专业术语
            features.append(1 if (操作词计数 + 药物词计数 + 监测词计数 + 并发症词计数 + 管理词计数) > 0 else 0)
            
            特征矩阵.append(features)
        
        return np.array(特征矩阵)
    
    def 提取高质量特征关键词(self, 文本列表):
        """提取高质量特征关键词"""
        特征矩阵 = []
        
        for 文本 in 文本列表:
            features = []
            
            if not 文本 or len(文本.strip()) == 0:
                特征矩阵.append([0] * len(self.高质量特征关键词))
                continue
            
            for feature, keywords in self.高质量特征关键词.items():
                关键词计数 = sum(1 for keyword in keywords if keyword in 文本)
                features.append(关键词计数)
            
            特征矩阵.append(features)
        
        return np.array(特征矩阵)
    
    def 提取NLP特征(self, 文本列表):
        """提取多种NLP特征"""
        特征矩阵 = []
        
        for 文本 in 文本列表:
            features = []
            
            # 1. 基本统计特征
            features.append(len(文本))
            features.append(len(文本.split()))
            
            # 2. 词汇多样性特征
            words = 文本.split()
            if len(words) > 0:
                unique_words = set(words)
                features.append(len(unique_words))
                features.append(len(unique_words) / len(words))
            else:
                features.extend([0, 0])
            
            # 3. 专业词汇特征
            专业词汇列表 = self.获取专业词汇列表()
            专业词计数 = sum(1 for word in words if word in 专业词汇列表)
            features.append(专业词计数)
            features.append(专业词计数 / max(1, len(words)))
            
            # 4. 情感特征
            positive_words = ["好", "优秀", "出色", "正确", "成功", "顺利"]
            negative_words = ["差", "错误", "失败", "不足", "问题", "缺点"]
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            features.extend([pos_count, neg_count, pos_count - neg_count])
            
            # 5. 句法特征
            标点符号 = ["。", "，", "；", "：", "！", "？"]
            punctuation_count = sum(1 for char in 文本 if char in 标点符号)
            features.append(punctuation_count)
            
            # 6. 数字特征
            digit_count = sum(1 for char in 文本 if char.isdigit())
            features.append(digit_count)
            
            # 7. 高质量关键词特征
            关键词特征 = []
            for feature, keywords in self.高质量特征关键词.items():
                关键词计数 = sum(1 for keyword in keywords if any(keyword in word for word in words))
                关键词特征.append(关键词计数)
            
            features.extend(关键词特征)
            
            特征矩阵.append(features)
        
        return np.array(特征矩阵)
    
    def 提取综合特征(self, 文本列表, 训练模式=True):
        """提取综合特征矩阵 - 修复版本"""
        特征列表 = []
        
        # 确保文本列表是列表类型
        if not isinstance(文本列表, list):
            文本列表 = list(文本列表)
        
        n_samples = len(文本列表)
        print(f"处理样本数: {n_samples}")
        
        # 1. TF-IDF特征
        X_tfidf = self.提取文本特征(文本列表, 训练模式=训练模式)
        if X_tfidf.shape[1] > 0:
            print(f"TF-IDF特征维度: {X_tfidf.shape}")
            # 确保形状正确
            if X_tfidf.shape[0] == n_samples:
                特征列表.append(X_tfidf)
            else:
                print(f"警告: TF-IDF特征样本数不匹配 ({X_tfidf.shape[0]} != {n_samples})")
        else:
            print("警告: TF-IDF特征为空")
        
        # 2. 增强的具体性矫正性特征
        X_enhanced = self.增强_具体性矫正性特征(文本列表)
        if X_enhanced.shape[1] > 0:
            print(f"增强特征维度: {X_enhanced.shape}")
            if X_enhanced.shape[0] == n_samples:
                特征列表.append(X_enhanced)
            else:
                print(f"警告: 增强特征样本数不匹配 ({X_enhanced.shape[0]} != {n_samples})")
        else:
            print("警告: 增强特征为空")
        
        # 3. 麻醉专业特征
        X_anesthesia = self.提取麻醉专业特征(文本列表)
        if X_anesthesia.shape[1] > 0:
            print(f"麻醉专业特征维度: {X_anesthesia.shape}")
            if X_anesthesia.shape[0] == n_samples:
                特征列表.append(X_anesthesia)
            else:
                print(f"警告: 麻醉专业特征样本数不匹配 ({X_anesthesia.shape[0]} != {n_samples})")
        else:
            print("警告: 麻醉专业特征为空")
        
        # 4. NLP特征
        X_nlp = self.提取NLP特征(文本列表)
        if X_nlp.shape[1] > 0:
            print(f"NLP特征维度: {X_nlp.shape}")
            if X_nlp.shape[0] == n_samples:
                特征列表.append(X_nlp)
            else:
                print(f"警告: NLP特征样本数不匹配 ({X_nlp.shape[0]} != {n_samples})")
        else:
            print("警告: NLP特征为空")
        
        if len(特征列表) == 0:
            print("错误: 所有特征提取都失败")
            return np.zeros((n_samples, 0))
        
        # 检查所有特征矩阵是否有相同的样本数
        for i, feat in enumerate(特征列表):
            if feat.shape[0] != n_samples:
                print(f"特征矩阵 {i} 样本数不匹配: {feat.shape[0]} != {n_samples}")
                # 调整维度
                if feat.shape[0] < n_samples:
                    # 补充零行
                    补零 = np.zeros((n_samples - feat.shape[0], feat.shape[1]))
                    feat = np.vstack([feat, 补零])
                else:
                    # 截断多余行
                    feat = feat[:n_samples, :]
                特征列表[i] = feat
        
        # 合并所有特征
        try:
            X_combined = np.hstack(特征列表)
            print(f"合并后总特征维度: {X_combined.shape}")
        except Exception as e:
            print(f"合并特征时出错: {e}")
            print("各特征维度:")
            for i, feat in enumerate(特征列表):
                print(f"  特征{i}: {feat.shape}")
            
            # 如果hstack失败，尝试手动合并
            # 确保所有特征有相同的维度
            max_features = max([feat.shape[1] for feat in 特征列表])
            aligned_features = []
            
            for feat in 特征列表:
                if feat.shape[1] < max_features:
                    # 补零
                    补零 = np.zeros((feat.shape[0], max_features - feat.shape[1]))
                    feat_aligned = np.hstack([feat, 补零])
                else:
                    feat_aligned = feat[:, :max_features]  # 截断
                aligned_features.append(feat_aligned)
            
            X_combined = np.hstack(aligned_features)
            print(f"手动对齐后维度: {X_combined.shape}")
        
        return X_combined
    
    def 获取专业词汇列表(self):
        """获取专业词汇列表"""
        try:
            with open(self.专业词库文件, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f.readlines()]
        except:
            return []
    
    def 验证数据格式(self):
        """验证输入数据是否符合新标准要求"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择数据文件进行格式验证",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            data = pd.read_excel(file_path)
            
            # 检查必要列
            必要列 = ['反馈文本', '相关性', '具体性', '矫正性']
            缺少列 = [col for col in 必要列 if col not in data.columns]
            
            if 缺少列:
                messagebox.showerror("格式错误", f"缺少以下必要列: {', '.join(缺少列)}")
                return
            
            # 尝试计算效用标签
            标签 = self.计算效用标签(data)
            if 标签 is not None:
                categories = ['无关', '无效', '中等', '有效']
                distribution = "\n".join([f"{cat}: {(标签==i).sum()}条" for i, cat in enumerate(categories)])
                
                messagebox.showinfo("数据验证结果", 
                    f"数据验证成功！\n\n总记录数: {len(data)}\n\n四级分类分布:\n{distribution}")
                
                return data
            else:
                messagebox.showerror("验证失败", "无法计算效用标签，请检查数据格式")
                
        except Exception as e:
            messagebox.showerror("验证错误", f"数据验证过程中出错: {str(e)}")
    
    def 数据预处理(self):
        """数据预处理：加载数据并移除空白反馈"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择训练数据Excel文件",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            data = pd.read_excel(file_path)
            
            if '反馈文本' not in data.columns:
                messagebox.showerror("错误", "Excel文件中必须包含'反馈文本'列")
                return
            
            # 移除空白反馈
            原始数量 = len(data)
            data = data.dropna(subset=['反馈文本'])
            data = data[data['反馈文本'].astype(str).str.strip() != '']
            处理后数量 = len(data)
            
            if 处理后数量 == 0:
                messagebox.showerror("错误", "数据预处理后没有有效的反馈文本")
                return
            
            self.进度标签.config(text="正在预处理文本...")
            self.root.update()
            
            data['处理文本'] = data['反馈文本'].apply(self.文本预处理)
            
            有效文本数量 = (data['处理文本'] != "").sum()
            if 有效文本数量 < 20:
                messagebox.showerror("错误", f"有效文本数量不足: {有效文本数量}，至少需要20个有效文本")
                return
            
            # 保存预处理数据
            self.预处理数据 = data
            self.进度标签.config(text=f"数据预处理完成！移除空白反馈: {原始数量 - 处理后数量}个，剩余: {处理后数量}个，有效文本: {有效文本数量}个")
            
        except Exception as e:
            messagebox.showerror("错误", f"数据预处理过程中出错: {str(e)}")
            self.进度标签.config(text="数据预处理失败")
    
    def 计算效用标签(self, data):
        """根据新的四级标准计算效用标签"""
        try:
            # 查找三个关键维度列
            相关性列 = None
            具体性列 = None
            矫正性列 = None
            
            for col in data.columns:
                if '相关性' in col or 'relevant' in col.lower():
                    相关性列 = col
                elif '具体性' in col or 'specific' in col.lower():
                    具体性列 = col
                elif '矫正性' in col or 'corrective' in col.lower():
                    矫正性列 = col
            
            if 相关性列 is None or 具体性列 is None or 矫正性列 is None:
                messagebox.showerror("错误", "数据中缺少相关性、具体性或矫正性列")
                return None
            
            print(f"使用列: 相关性={相关性列}, 具体性={具体性列}, 矫正性={矫正性列}")
            
            # 转换为布尔类型
            def 转换为布尔(series):
                if series.dtype == 'object':
                    # 处理各种可能的字符串表示
                    return series.astype(str).str.lower().isin(['true', '是', 'yes', '1', 't', '真', 'yes ', 'true ', '对', '正确'])
                else:
                    return series.astype(bool)
            
            # 计算四个类别
            相关性 = 转换为布尔(data[相关性列])
            具体性 = 转换为布尔(data[具体性列])
            矫正性 = 转换为布尔(data[矫正性列])
            
            # 应用新标准
            有效 = 相关性 & 具体性 & 矫正性
            中等 = 相关性 & (具体性 | 矫正性) & ~有效
            无效 = 相关性 & ~具体性 & ~矫正性
            无关 = ~相关性
            
            # 创建标签映射：有效=3, 中等=2, 无效=1, 无关=0
            效用标签 = np.zeros(len(data), dtype=int)
            效用标签[中等] = 2
            效用标签[有效] = 3
            效用标签[无效] = 1
            # 无关保持为0
            
            # 打印详细分布
            categories = ['无关', '无效', '中等', '有效']
            print("\n=== 四级分类详细分布 ===")
            for i, cat in enumerate(categories):
                count = (效用标签 == i).sum()
                percentage = count/len(data)*100
                print(f"{cat}: {count}条 ({percentage:.1f}%)")
            
            # 打印逻辑组合分布
            print("\n=== 逻辑组合分布 ===")
            print(f"相关性 & 具体性 & 矫正性 (有效): {有效.sum()}条")
            print(f"相关性 & (具体性 | 矫正性) & ~有效 (中等): {中等.sum()}条")
            print(f"相关性 & ~具体性 & ~矫正性 (无效): {无效.sum()}条")
            print(f"~相关性 (无关): {无关.sum()}条")
            
            return 效用标签
            
        except Exception as e:
            print(f"计算效用标签时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def 计算置信区间(self, scores, confidence=0.95):
        """计算置信区间"""
        if len(scores) == 0:
            return 0.0, 0.0
            
        sorted_scores = np.array(scores)
        sorted_scores.sort()
        
        lower_percentile = (1.0 - confidence) / 2.0 * 100
        upper_percentile = (confidence + (1.0 - confidence) / 2.0) * 100
        
        lower = max(0.0, np.percentile(sorted_scores, lower_percentile))
        upper = min(1.0, np.percentile(sorted_scores, upper_percentile))
        
        return lower, upper
    
    def 训练测试数据(self):
        """训练和测试效用模型（四级分类）"""
        try:
            if self.预处理数据 is None:
                messagebox.showwarning("警告", "请先进行数据预处理")
                return
            
            data = self.预处理数据.copy()
            
            self.进度标签.config(text="使用四级分类标准...")
            y_utility = self.计算效用标签(data)
            if y_utility is None:
                return
            
            n_classes = 4
            
            # 检查标签多样性
            unique_classes = np.unique(y_utility)
            if len(unique_classes) < n_classes:
                messagebox.showwarning("警告", 
                    f"目标变量缺乏多样性，需要至少 {n_classes} 个类别的样本\n"
                    f"当前类别数: {len(unique_classes)}")
                
                # 显示当前分布
                categories = ['无关', '无效', '中等', '有效']
                distribution = "\n".join([f"{categories[i] if i < len(categories) else f'类别{i}'}: {(y_utility==i).sum()}条" 
                                        for i in unique_classes])
                messagebox.showinfo("类别分布", f"当前类别分布:\n{distribution}")
                return
            
            # 提取综合特征
            self.进度标签.config(text="正在提取综合特征...")
            self.root.update()
            
            X = self.提取综合特征(data['处理文本'].tolist(), 训练模式=True)
            
            if X.shape[1] == 0:
                messagebox.showerror("错误", "无法提取任何特征，请检查文本数据")
                return
            
            self.训练特征维度 = X.shape[1]
            print(f"\n=== 特征信息 ===")
            print(f"特征矩阵总形状: {X.shape}")
            print(f"训练特征维度: {self.训练特征维度}")
            
            # 7:3分层划分训练测试集
            X_train, X_test, yu_train, yu_test = train_test_split(
                X, y_utility, test_size=0.3, random_state=42, 
                stratify=y_utility
            )
            
            print(f"\n=== 数据集划分 ===")
            print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
            
            # 使用SMOTE处理不平衡数据
            self.进度标签.config(text="正在处理数据不平衡...")
            self.root.update()
            
            try:
                smote = SMOTE(random_state=42)
                X_train_resampled, yu_train_resampled = smote.fit_resample(X_train, yu_train)
                print(f"SMOTE处理后训练集形状: {X_train_resampled.shape}")
            except Exception as e:
                print(f"SMOTE处理失败: {e}，使用原始数据")
                X_train_resampled, yu_train_resampled = X_train, yu_train
            
            # 特征标准化
            self.标准化器 = StandardScaler()
            try:
                X_train_scaled = self.标准化器.fit_transform(X_train_resampled)
                X_test_scaled = self.标准化器.transform(X_test)
            except Exception as e:
                print(f"标准化失败: {e}")
                X_train_scaled, X_test_scaled = X_train_resampled, X_test
            
            # 使用增强的模型参数
            models_params = self.获取增强模型参数()
            
            结果 = []
            self.模型字典 = {}
            
            for name, mp in models_params.items():
                self.进度标签.config(text=f"训练{name}模型...")
                self.root.update()
                
                try:
                    print(f"\n开始训练{name}...")
                    
                    if mp['params']:  # 如果有参数网格，使用网格搜索
                        grid_search = GridSearchCV(
                            mp['model'], 
                            mp['params'], 
                            cv=5,
                            scoring='f1_weighted',
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        grid_search.fit(X_train_scaled, yu_train_resampled)
                        
                        # 获取最佳模型
                        best_model = grid_search.best_estimator_
                        最佳参数 = grid_search.best_params_
                    else:
                        # 直接训练
                        best_model = mp['model']
                        best_model.fit(X_train_scaled, yu_train_resampled)
                        最佳参数 = "无参数调优"
                    
                    # 预测
                    yu_pred = best_model.predict(X_test_scaled)
                    
                    # 计算四级分类指标
                    u_accuracy = accuracy_score(yu_test, yu_pred)
                    u_precision = precision_score(yu_test, yu_pred, average='weighted', zero_division=0)
                    u_recall = recall_score(yu_test, yu_pred, average='weighted', zero_division=0)
                    u_f1 = f1_score(yu_test, yu_pred, average='weighted', zero_division=0)
                    
                    结果.append({
                        '算法': name,
                        '分类方式': '四级分类',
                        '准确率': round(u_accuracy, 3),
                        '精确率': round(u_precision, 3),
                        '召回率': round(u_recall, 3),
                        'F1分数': round(u_f1, 3),
                        '最佳参数': str(最佳参数)
                    })
                    
                    # 保存模型
                    self.模型字典[name] = {
                        '效用模型': best_model,
                        '性能': {'准确率': u_accuracy, 'F1': u_f1},
                        '最佳参数': 最佳参数,
                    }
                    
                    print(f"{name}训练成功，F1分数: {u_f1:.4f}")
                    
                except Exception as e:
                    print(f"训练{name}时出错: {e}")
                    continue
            
            # 选择最佳模型
            if 结果:
                best_model_info = max(结果, key=lambda x: x['F1分数'])
                
                self.效用模型 = self.模型字典[best_model_info['算法']]['效用模型']
                self.最佳算法 = f"{best_model_info['算法']}(最佳)"
                self.最佳参数 = self.模型字典[best_model_info['算法']]['最佳参数']
                
                # 保存性能结果
                self.训练性能结果 = pd.DataFrame(结果)
                
                # 显示结果
                self.显示训练结果(结果)
                self.进度标签.config(text=f"模型训练完成！分类方式: 四级分类, 最佳模型: {self.最佳算法}")
                
                # 计算并保存混淆矩阵
                self.混淆矩阵 = confusion_matrix(yu_test, yu_pred)
                
                # 自动保存性能结果
                self.保存性能结果()
                
                # 显示分类报告
                categories = ['无关', '无效', '中等', '有效']
                
                print(f"\n=== 分类报告 ===")
                print(classification_report(yu_test, yu_pred, target_names=categories))
                
                # 显示混淆矩阵
                print(f"\n=== 混淆矩阵 ===")
                print(self.混淆矩阵)
                
                # 显示最佳模型信息
                messagebox.showinfo("训练完成", 
                    f"模型训练完成！\n\n最佳模型: {self.最佳算法}\n"
                    f"F1分数: {best_model_info['F1分数']:.3f}\n"
                    f"准确率: {best_model_info['准确率']:.3f}\n\n"
                    f"已保存训练结果和模型。")
                
            else:
                messagebox.showerror("错误", "所有模型训练失败，请检查数据质量")
            
        except Exception as e:
            messagebox.showerror("错误", f"训练过程中出错: {str(e)}")
            self.进度标签.config(text="训练失败")
    
    def 获取增强模型参数(self):
        """获取增强的模型参数"""
        return {
            '梯度提升机': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5]
                }
            },
            '随机森林': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            '逻辑回归': {
                'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', multi_class='multinomial'),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'newton-cg']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(
                    random_state=42,
                    objective='multi:softprob',
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(
                    random_state=42,
                    class_weight='balanced',
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [100, 200],
                    'num_leaves': [31, 50, 100],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [5, 10]
                }
            }
        }
    
    def 显示训练结果(self, 结果):
        """显示训练结果"""
        for item in self.结果树.get_children():
            self.结果树.delete(item)
        
        for res in 结果:
            self.结果树.insert("", "end", values=(
                res['算法'], res['分类方式'], res['准确率'], res['精确率'], 
                res['召回率'], res['F1分数'], res['最佳参数']))
    
    def 保存性能结果(self):
        """保存算法性能结果到Excel文件"""
        try:
            if self.训练性能结果 is not None:
                时间戳 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                文件名 = f"效用模型性能结果_{时间戳}.xlsx"
                
                # 添加训练信息
                if hasattr(self, '预处理数据'):
                    效用标签 = self.计算效用标签(self.预处理数据)
                    if 效用标签 is not None:
                        categories = ['无关', '无效', '中等', '有效']
                        分布数据 = {cat: (效用标签==i).sum() for i, cat in enumerate(categories)}
                    else:
                        分布数据 = {}
                else:
                    分布数据 = {}
                
                性能报告 = pd.DataFrame({
                    '训练时间': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    '分类方式': ['四级分类'],
                    '训练模式': ['稳定模式'],
                    '最佳算法': [self.最佳算法 if hasattr(self, '最佳算法') else '未知'],
                    '最佳参数': [str(self.最佳参数) if hasattr(self, '最佳参数') else '未知'],
                    '特征维度': [self.训练特征维度 if hasattr(self, '训练特征维度') else 0],
                    '总样本数': [len(self.预处理数据) if hasattr(self, '预处理数据') else 0],
                    '四级分类-无关': [分布数据.get('无关', 0)],
                    '四级分类-无效': [分布数据.get('无效', 0)],
                    '四级分类-中等': [分布数据.get('中等', 0)],
                    '四级分类-有效': [分布数据.get('有效', 0)],
                    '使用特征': ['TF-IDF + NLP特征 + 增强特征 + 麻醉专业特征'],
                    'BERT状态': ['未启用']
                })
                
                with pd.ExcelWriter(文件名, engine='openpyxl') as writer:
                    性能报告.to_excel(writer, sheet_name='训练信息', index=False)
                    self.训练性能结果.to_excel(writer, sheet_name='算法性能', index=False)
                    
                    # 添加混淆矩阵
                    if self.混淆矩阵 is not None:
                        混淆矩阵_df = pd.DataFrame(
                            self.混淆矩阵,
                            index=['实际-无关', '实际-无效', '实际-中等', '实际-有效'],
                            columns=['预测-无关', '预测-无效', '预测-中等', '预测-有效']
                        )
                        混淆矩阵_df.to_excel(writer, sheet_name='混淆矩阵')
                    
                    # 添加分类标准说明
                    标准说明 = pd.DataFrame({
                        '分类等级': ['有效', '中等', '无效', '无关'],
                        '逻辑条件': [
                            '相关性 AND 具体性 AND 矫正性',
                            '相关性 AND (具体性 OR 矫正性)',
                            '相关性 NOT (具体性 AND 矫正性)',
                            'NOT 相关性'
                        ],
                        '说明': [
                            '同时具备相关性、具体性和矫正性，是最理想的反馈',
                            '具备相关性和至少一项具体性或矫正性',
                            '具备相关性但不具备具体性和矫正性',
                            '内容与表现无关'
                        ]
                    })
                    标准说明.to_excel(writer, sheet_name='分类标准', index=False)
                
                print(f"性能结果已保存到: {文件名}")
                return 文件名
            else:
                print("没有性能结果可保存")
                return None
        except Exception as e:
            print(f"保存性能结果失败: {e}")
            return None
    
    def 保存模型(self):
        """保存训练好的模型"""
        try:
            if self.效用模型 is not None:
                时间戳 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 保存模型和所有相关组件
                模型数据 = {
                    '效用模型': self.效用模型,
                    '分类方式': '四级分类',
                    '最佳算法': self.最佳算法 if hasattr(self, '最佳算法') else '未知',
                    '训练特征维度': self.训练特征维度 if hasattr(self, '训练特征维度') else 0,
                    '保存时间': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    '版本': '稳定版v1.0'
                }
                
                with open(f'效用模型_{时间戳}.pkl', 'wb') as f:
                    pickle.dump(模型数据, f)
                
                # 保存向量化器
                if self.向量化器 is not None:
                    with open(f'向量化器_{时间戳}.pkl', 'wb') as f:
                        pickle.dump(self.向量化器, f)
                
                # 保存标准化器
                if self.标准化器 is not None:
                    with open(f'标准化器_{时间戳}.pkl', 'wb') as f:
                        pickle.dump(self.标准化器, f)
                
                messagebox.showinfo("成功", f"模型保存成功！\n\n保存文件:\n"
                                      f"• 效用模型_{时间戳}.pkl\n"
                                      f"• 向量化器_{时间戳}.pkl\n"
                                      f"• 标准化器_{时间戳}.pkl")
            else:
                messagebox.showwarning("警告", "没有训练好的模型可保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存模型失败: {str(e)}")
    
    def 加载模型(self):
        """加载已训练的模型"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择要加载的模型文件",
                filetypes=[("模型文件", "*.pkl"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
            
            with open(file_path, 'rb') as f:
                模型数据 = pickle.load(f)
            
            if isinstance(模型数据, dict) and '效用模型' in 模型数据:
                self.效用模型 = 模型数据['效用模型']
                self.最佳算法 = 模型数据.get('最佳算法', '未知')
                self.训练特征维度 = 模型数据.get('训练特征维度', 0)
                print(f"加载模型: 分类方式=四级分类, 算法={self.最佳算法}, 特征维度={self.训练特征维度}")
            else:
                self.效用模型 = 模型数据
            
            # 尝试加载其他组件
            模型目录 = os.path.dirname(file_path)
            模型前缀 = os.path.basename(file_path).replace('效用模型_', '').replace('.pkl', '')
            
            向量化器文件 = os.path.join(模型目录, f'向量化器_{模型前缀}.pkl')
            标准化器文件 = os.path.join(模型目录, f'标准化器_{模型前缀}.pkl')
            
            if os.path.exists(向量化器文件):
                with open(向量化器文件, 'rb') as f:
                    self.向量化器 = pickle.load(f)
                print(f"加载向量化器: 词汇量={len(self.向量化器.vocabulary_) if hasattr(self.向量化器, 'vocabulary_') else '未知'}")
            
            if os.path.exists(标准化器文件):
                with open(标准化器文件, 'rb') as f:
                    self.标准化器 = pickle.load(f)
                print("加载标准化器成功")
            
            messagebox.showinfo("成功", f"模型加载成功！\n\n分类方式: 四级分类\n特征维度: {self.训练特征维度}")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")
    
    def 测试新数据(self):
        """测试新的反馈数据并自动保存结果"""
        try:
            if self.效用模型 is None:
                messagebox.showwarning("警告", "请先训练或加载模型")
                return
            
            # 检查特征提取器是否已加载
            if self.向量化器 is None:
                messagebox.showwarning("警告", "请先训练模型或加载包含特征提取器的模型文件")
                return
            
            file_path = filedialog.askopenfilename(
                title="选择要测试的Excel文件",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            data = pd.read_excel(file_path)
            
            if '反馈文本' not in data.columns:
                messagebox.showerror("错误", "Excel文件中必须包含'反馈文本'列")
                return
            
            # 检查是否有三个关键维度列
            has_dimensions = all(col in data.columns for col in ['相关性', '具体性', '矫正性'])
            
            self.进度标签.config(text="正在测试数据...")
            self.root.update()
            
            # 预处理文本并识别空白反馈
            data['处理文本'] = data['反馈文本'].apply(self.文本预处理)
            
            # 识别空白反馈
            空白反馈掩码 = data['处理文本'] == ""
            空白数量 = 空白反馈掩码.sum()
            
            # 计算真实标签（如果存在维度数据）
            if has_dimensions:
                y_true_all = self.计算效用标签(data)
                
                if y_true_all is not None:
                    y_true_nonblank = y_true_all[~空白反馈掩码]
                    有真实标签 = len(y_true_nonblank) > 0
                else:
                    有真实标签 = False
                    y_true_nonblank = None
            else:
                有真实标签 = False
                y_true_nonblank = None
            
            # 为非空白反馈提取特征
            if 空白数量 < len(data):
                非空白数据 = data[~空白反馈掩码]
                
                # 提取综合特征（测试模式）
                X = self.提取综合特征(非空白数据['处理文本'].tolist(), 训练模式=False)
                print(f"测试特征维度: {X.shape}")
                
                if X.shape[1] > 0:
                    # 检查特征维度是否与训练时一致
                    if hasattr(self, '训练特征维度') and self.训练特征维度 > 0:
                        if X.shape[1] != self.训练特征维度:
                            print(f"警告：特征维度不匹配！训练时: {self.训练特征维度}, 测试时: {X.shape[1]}")
                            # 调整维度以匹配训练时
                            if X.shape[1] < self.训练特征维度:
                                # 补零
                                补零数量 = self.训练特征维度 - X.shape[1]
                                补零矩阵 = np.zeros((X.shape[0], 补零数量))
                                X = np.hstack([X, 补零矩阵])
                                print(f"已补零，调整后维度: {X.shape}")
                            else:
                                # 截断
                                X = X[:, :self.训练特征维度]
                                print(f"已截断，调整后维度: {X.shape}")
                    
                    # 标准化
                    if self.标准化器 is not None:
                        try:
                            X = self.标准化器.transform(X)
                        except Exception as e:
                            print(f"标准化失败: {e}")
                            # 如果标准化失败，使用原始特征
                    
                    # 预测（四级分类）
                    预测结果 = self.效用模型.predict(X)
                    if hasattr(self.效用模型, 'predict_proba'):
                        概率 = self.效用模型.predict_proba(X)
                        # 取最大概率
                        预测概率 = np.max(概率, axis=1)
                    else:
                        预测概率 = np.ones(len(预测结果))
                    
                    # 将预测结果放回原数据
                    data.loc[~空白反馈掩码, '效用预测'] = 预测结果
                    data.loc[~空白反馈掩码, '效用概率'] = 预测概率
                    
                    # 计算准确率（如果存在真实标签）
                    准确率 = None
                    准确率_ci_lower = None
                    准确率_ci_upper = None
                    
                    if 有真实标签 and y_true_nonblank is not None:
                        if len(y_true_nonblank) == len(预测结果):
                            准确率 = accuracy_score(y_true_nonblank, 预测结果)
                            print(f"去掉空白后的分类准确率: {准确率:.4f}")
                            
                            # 计算置信区间
                            bootstrapped_acc = []
                            for i in range(100):
                                indices = np.random.randint(0, len(X), len(X))
                                X_boot = X[indices]
                                y_true_boot = y_true_nonblank.iloc[indices] if hasattr(y_true_nonblank, 'iloc') else y_true_nonblank[indices]
                                y_pred_boot = self.效用模型.predict(X_boot)
                                acc = accuracy_score(y_true_boot, y_pred_boot)
                                bootstrapped_acc.append(acc)
                            
                            准确率_ci_lower, 准确率_ci_upper = self.计算置信区间(bootstrapped_acc)
                            print(f"分类准确率95%CI: [{准确率_ci_lower:.4f}, {准确率_ci_upper:.4f}]")
                            
                            # 计算混淆矩阵
                            self.混淆矩阵 = confusion_matrix(y_true_nonblank, 预测结果)
                            print(f"混淆矩阵:\n{self.混淆矩阵}")
                        else:
                            print(f"警告：标签长度不匹配，真实标签: {len(y_true_nonblank)}, 预测结果: {len(预测结果)}")
                else:
                    # 如果没有有效特征，标记为最低类别（无关）
                    data.loc[~空白反馈掩码, '效用预测'] = 0
                    data.loc[~空白反馈掩码, '效用概率'] = 0.0
            
            # 空白反馈直接标记为无关（0）
            data.loc[空白反馈掩码, '效用预测'] = 0
            data.loc[空白反馈掩码, '效用概率'] = 0.0
            
            # 添加四级分类标签
            def 获取四级标签(预测值):
                if 预测值 == 0:
                    return '无关'
                elif 预测值 == 1:
                    return '无效'
                elif 预测值 == 2:
                    return '中等'
                elif 预测值 == 3:
                    return '有效'
                else:
                    return f'未知({预测值})'
            
            data['效用标签'] = data['效用预测'].apply(获取四级标签)
            
            # 如果有维度数据，添加详细信息
            if has_dimensions:
                data['相关性'] = data['相关性'].astype(str)
                data['具体性'] = data['具体性'].astype(str)
                data['矫正性'] = data['矫正性'].astype(str)
            
            # 显示结果
            self.显示分析结果(data, has_dimensions)
            
            # 保存结果
            self.当前结果 = data
            
            # 自动保存测试结果
            保存路径 = self.自动保存测试结果(data, has_dimensions, 有真实标签, 准确率, 
                                        准确率_ci_lower, 准确率_ci_upper)
            
            self.进度标签.config(text=f"测试完成！分类方式: 四级分类, 空白反馈: {空白数量}个")
            
            # 显示统计信息
            categories = ['无关', '无效', '中等', '有效']
            stats = []
            for i, cat in enumerate(categories):
                count = (data['效用预测'] == i).sum()
                percentage = count/len(data)*100
                stats.append(f"{cat}: {percentage:.1f}%")
            统计文本 = " | ".join(stats)
            
            self.统计标签.config(text=f"{统计文本} | 空白反馈: {空白数量}个")
            
            # 显示准确率信息
            if 有真实标签 and 准确率 is not None:
                详细信息 = ""
                if self.混淆矩阵 is not None:
                    categories = ['无关', '无效', '中等', '有效']
                    详细信息 = "\n\n混淆矩阵:\n"
                    for i in range(4):
                        for j in range(4):
                            详细信息 += f"{categories[i]}->{categories[j]}: {self.混淆矩阵[i, j]}  "
                        详细信息 += "\n"
                
                messagebox.showinfo("测试结果", 
                    f"分类方式: 四级分类\n"
                    f"总样本数: {len(data)}条\n"
                    f"空白反馈: {空白数量}条\n"
                    f"非空白样本: {len(data) - 空白数量}条\n\n"
                    f"去掉空白后的分类准确率: {准确率:.3f}\n"
                    f"95%置信区间: [{准确率_ci_lower:.3f}, {准确率_ci_upper:.3f}]\n"
                    f"{详细信息}"
                    f"结果已保存到:\n{保存路径}")
            else:
                messagebox.showinfo("测试结果", 
                    f"分类方式: 四级分类\n"
                    f"总样本数: {len(data)}条\n"
                    f"空白反馈: {空白数量}条\n\n"
                    f"结果已保存到:\n{保存路径}")
            
        except Exception as e:
            messagebox.showerror("错误", f"测试过程中出错: {str(e)}")
            self.进度标签.config(text="测试失败")
    
    def 自动保存测试结果(self, data, has_dimensions, 有真实标签, 准确率, 
                      准确率_ci_lower, 准确率_ci_upper):
        """自动保存测试结果到Excel文件"""
        try:
            时间戳 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            文件名 = f"效用测试结果_{时间戳}.xlsx"
            
            # 创建详细的测试报告
            测试报告数据 = {
                '测试时间': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                '分类方式': ['四级分类'],
                '总样本数': [len(data)],
                '空白反馈': [(data['处理文本'] == "").sum()],
                '使用模型': [self.最佳算法 if hasattr(self, '最佳算法') else "已加载模型"],
                '模型参数': [str(self.最佳参数) if hasattr(self, '最佳参数') else "默认参数"],
                '特征维度': [self.训练特征维度 if hasattr(self, '训练特征维度') else "未知"],
                '使用BERT特征': ['否'],
                '特征组合': ['TF-IDF + NLP特征 + 增强特征 + 麻醉专业特征']
            }
            
            # 添加分类分布
            categories = ['无关', '无效', '中等', '有效']
            for i, cat in enumerate(categories):
                count = (data['效用预测'] == i).sum()
                测试报告数据[f'{cat}数量'] = [count]
                测试报告数据[f'{cat}比例'] = [f"{count/len(data)*100:.1f}%"]
            
            # 添加准确率信息
            if 有真实标签 and 准确率 is not None:
                测试报告数据['分类准确率'] = [准确率]
                测试报告数据['准确率95%CI下限'] = [准确率_ci_lower]
                测试报告数据['准确率95%CI上限'] = [准确率_ci_upper]
            
            测试报告数据['分类标准说明'] = ['相关性 AND 具体性 AND 矫正性 = 有效\n相关性 AND (具体性 OR 矫正性) = 中等\n相关性 NOT (具体性 AND 矫正性) = 无效\nNOT 相关性 = 无关']
            
            测试报告 = pd.DataFrame(测试报告数据)
            
            # 保存详细结果
            with pd.ExcelWriter(文件名, engine='openpyxl') as writer:
                测试报告.to_excel(writer, sheet_name='测试摘要', index=False)
                
                # 确定输出列
                输出列 = ['反馈文本', '处理文本', '效用标签', '效用概率']
                if has_dimensions:
                    输出列.extend(['相关性', '具体性', '矫正性'])
                
                data[输出列].to_excel(writer, sheet_name='详细结果', index=False)
                
                # 添加分类标准表
                分类标准 = pd.DataFrame({
                    '分类等级': ['有效', '中等', '无效', '无关'],
                    '逻辑条件': [
                        '相关性 AND 具体性 AND 矫正性',
                        '相关性 AND (具体性 OR 矫正性)',
                        '相关性 NOT (具体性 AND 矫正性)',
                        'NOT 相关性'
                    ],
                    '特征要求': [
                        '同时具备相关性、具体性和矫正性',
                        '具备相关性和至少一项具体性或矫正性',
                        '具备相关性但不具备具体性和矫正性',
                        '内容与表现无关'
                    ],
                    '教学价值': [
                        '高 - 最理想的反馈，直接指导改进',
                        '中 - 有用的反馈，但不够全面',
                        '低 - 空洞的反馈，缺乏指导性',
                        '无 - 无教学价值'
                    ]
                })
                分类标准.to_excel(writer, sheet_name='分类标准', index=False)
                
                # 添加混淆矩阵
                if self.混淆矩阵 is not None:
                    混淆矩阵_df = pd.DataFrame(
                        self.混淆矩阵,
                        index=['实际-无关', '实际-无效', '实际-中等', '实际-有效'],
                        columns=['预测-无关', '预测-无效', '预测-中等', '预测-有效']
                    )
                    混淆矩阵_df.to_excel(writer, sheet_name='混淆矩阵')
            
            print(f"测试结果已保存到: {文件名}")
            return 文件名
        except Exception as e:
            print(f"自动保存测试结果失败: {e}")
            return "保存失败"
    
    def 显示分析结果(self, data, has_dimensions=False):
        """显示分析结果"""
        for item in self.分析结果树.get_children():
            self.分析结果树.delete(item)
        
        # 更新列
        if has_dimensions:
            columns = ("序号", "反馈文本", "效用标签", "效用概率", "相关性", "具体性", "矫正性")
        else:
            columns = ("序号", "反馈文本", "效用标签", "效用概率")
        
        self.分析结果树["columns"] = columns
        for col in columns:
            self.分析结果树.heading(col, text=col)
            self.分析结果树.column(col, width=100 if col != "反馈文本" else 400)
        
        for idx, row in data.iterrows():
            反馈文本 = str(row['反馈文本'])
            显示文本 = 反馈文本[:60] + "..." if len(反馈文本) > 60 else 反馈文本
            
            if row['处理文本'] == "":
                显示文本 = "[空白反馈] " + 显示文本
            
            if has_dimensions:
                values = (
                    idx + 1,
                    显示文本,
                    row.get('效用标签', '未知'),
                    f"{row.get('效用概率', 0):.3f}",
                    row.get('相关性', ''),
                    row.get('具体性', ''),
                    row.get('矫正性', '')
                )
            else:
                values = (
                    idx + 1,
                    显示文本,
                    row.get('效用标签', '未知'),
                    f"{row.get('效用概率', 0):.3f}"
                )
            
            self.分析结果树.insert("", "end", values=values)
    
    def 显示特征重要性(self):
        """显示特征重要性分析"""
        if not hasattr(self.效用模型, 'feature_importances_'):
            messagebox.showinfo("信息", "当前模型不支持特征重要性分析")
            return
        
        重要性窗口 = tk.Toplevel(self.root)
        重要性窗口.title("特征重要性分析")
        重要性窗口.geometry("800x600")
        
        # 获取特征重要性
        重要性 = self.效用模型.feature_importances_
        
        top_n = min(30, len(重要性))
        indices = np.argsort(重要性)[::-1][:top_n]
        
        # 创建DataFrame显示
        top_features = pd.DataFrame({
            '排名': range(1, top_n + 1),
            '特征重要性': [重要性[i] for i in indices],
            '特征索引': indices,
            '相对重要性%': [重要性[i]/重要性.sum()*100 for i in indices]
        })
        
        # 创建文本框显示
        文本框 = tk.Text(重要性窗口, wrap=tk.WORD, font=("宋体", 10))
        滚动条 = ttk.Scrollbar(重要性窗口, orient=tk.VERTICAL, command=文本框.yview)
        文本框.configure(yscrollcommand=滚动条.set)
        
        文本框.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        滚动条.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # 添加内容
        文本框.insert(tk.END, "Top 30 最重要特征:\n\n")
        文本框.insert(tk.END, top_features.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        文本框.insert(tk.END, "\n\n特征类型说明:\n")
        文本框.insert(tk.END, "1. TF-IDF特征: 文本向量化特征\n")
        文本框.insert(tk.END, "2. 增强特征: 具体性、矫正性等特征\n")
        文本框.insert(tk.END, "3. 专业特征: 麻醉专业相关特征\n")
        文本框.insert(tk.END, "4. NLP特征: 统计和句法特征\n")
        
        文本框.config(state=tk.DISABLED)
    
    def 显示混淆矩阵图(self):
        """显示混淆矩阵图"""
        if self.混淆矩阵 is None:
            messagebox.showinfo("信息", "没有可用的混淆矩阵数据")
            return
        
        # 创建图形窗口
        fig, ax = plt.subplots(figsize=(8, 6))
        
        categories = ['无关', '无效', '中等', '有效']
        
        # 创建热图
        sns.heatmap(self.混淆矩阵, annot=True, fmt='d', cmap='Blues',
                    xticklabels=categories, yticklabels=categories,
                    ax=ax)
        
        ax.set_title('混淆矩阵')
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        
        plt.tight_layout()
        plt.show()
    
    def 添加词汇界面(self):
        """打开添加专业词汇的界面"""
        词汇窗口 = tk.Toplevel(self.root)
        词汇窗口.title("添加麻醉专业词汇")
        词汇窗口.geometry("400x300")
        
        tk.Label(词汇窗口, text="输入要添加的专业词汇（每行一个）:").pack(pady=10)
        
        词汇文本框 = tk.Text(词汇窗口, height=10, width=40)
        词汇文本框.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        def 添加词汇():
            词汇列表 = 词汇文本框.get("1.0", tk.END).strip().split('\n')
            词汇列表 = [word.strip() for word in 词汇列表 if word.strip()]
            
            if 词汇列表:
                if self.添加专业词汇(词汇列表):
                    messagebox.showinfo("成功", f"成功添加 {len(词汇列表)} 个专业词汇")
                    词汇窗口.destroy()
                else:
                    messagebox.showerror("错误", "添加词汇失败")
            else:
                messagebox.showwarning("警告", "请输入要添加的词汇")
        
        tk.Button(词汇窗口, text="添加词汇", command=添加词汇).pack(pady=10)
    
    def 切换BERT模式(self):
        """切换BERT特征使用模式 - 当前版本不支持"""
        messagebox.showinfo("提示", "当前版本为稳定版，暂不支持BERT功能。\n\n如需使用BERT功能，请安装transformers库和bert模型。")
    
    def 创建界面(self):
        """创建主界面"""
        # 主框架
        主框架 = ttk.Frame(self.root)
        主框架.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 顶部按钮框架
        顶部按钮帧 = ttk.Frame(主框架)
        顶部按钮帧.pack(fill=tk.X, pady=5)
        
        ttk.Button(顶部按钮帧, text="验证数据格式", command=self.验证数据格式).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="数据预处理", command=self.数据预处理).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="训练测试数据", command=self.训练测试数据).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="保存模型", command=self.保存模型).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="加载模型", command=self.加载模型).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="测试新数据", command=self.测试新数据).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="添加专业词汇", command=self.添加词汇界面).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="特征重要性", command=self.显示特征重要性).pack(side=tk.LEFT, padx=5)
        ttk.Button(顶部按钮帧, text="混淆矩阵图", command=self.显示混淆矩阵图).pack(side=tk.LEFT, padx=5)
        
        # 版本说明框架
        版本说明帧 = ttk.Frame(主框架)
        版本说明帧.pack(fill=tk.X, pady=5)
        
        self.BERT模式标签 = ttk.Label(版本说明帧, 
                                    text="稳定版 - 不使用BERT，运行更快速",
                                    foreground="green")
        self.BERT模式标签.pack(side=tk.LEFT, padx=5)
        
        # 分类方式说明框架
        分类方式帧 = ttk.LabelFrame(主框架, text="分类方式")
        分类方式帧.pack(fill=tk.X, pady=5)
        
        ttk.Label(分类方式帧, text="当前使用四级分类（有效/中等/无效/无关）", 
                 font=("宋体", 10, "bold"), foreground="blue").pack(padx=10, pady=5)
        
        ttk.Label(分类方式帧, text="使用TF-IDF + NLP特征 + 增强特征 + 麻醉专业特征", 
                 font=("宋体", 9), foreground="green").pack(padx=10, pady=2)
        
        # 进度和统计标签
        self.进度标签 = ttk.Label(主框架, text="准备就绪")
        self.进度标签.pack(pady=2)
        
        self.统计标签 = ttk.Label(主框架, text="", foreground="blue")
        self.统计标签.pack(pady=2)
        
        # 训练结果框架
        训练结果帧 = ttk.LabelFrame(主框架, text="模型性能比较")
        训练结果帧.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 训练结果树状图
        columns = ("算法", "分类方式", "准确率", "精确率", "召回率", "F1分数", "最佳参数")
        self.结果树 = ttk.Treeview(训练结果帧, columns=columns, show="headings", height=6)
        
        column_widths = {
            "算法": 100,
            "分类方式": 80,
            "准确率": 60,
            "精确率": 60,
            "召回率": 60,
            "F1分数": 60,
            "最佳参数": 200
        }
        
        for col in columns:
            self.结果树.heading(col, text=col)
            self.结果树.column(col, width=column_widths.get(col, 100))
        
        # 滚动条
        滚动条 = ttk.Scrollbar(训练结果帧, orient=tk.VERTICAL, command=self.结果树.yview)
        self.结果树.configure(yscrollcommand=滚动条.set)
        
        self.结果树.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        滚动条.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 分析结果框架
        分析结果帧 = ttk.LabelFrame(主框架, text="详细分析结果")
        分析结果帧.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 分析结果树状图
        columns = ("序号", "反馈文本", "效用标签", "效用概率")
        self.分析结果树 = ttk.Treeview(分析结果帧, columns=columns, show="headings", height=12)
        
        column_widths = {
            "序号": 50,
            "反馈文本": 400,
            "效用标签": 100,
            "效用概率": 100
        }
        
        for col in columns:
            self.分析结果树.heading(col, text=col)
            self.分析结果树.column(col, width=column_widths.get(col, 100))
        
        # 滚动条
        分析滚动条 = ttk.Scrollbar(分析结果帧, orient=tk.VERTICAL, command=self.分析结果树.yview)
        self.分析结果树.configure(yscrollcommand=分析滚动条.set)
        
        self.分析结果树.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        分析滚动条.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 添加分类标准说明
        说明帧 = ttk.LabelFrame(主框架, text="四级分类标准说明")
        说明帧.pack(fill=tk.X, pady=5)
        
        说明文本 = (
            "1. 有效：相关性 AND 具体性 AND 矫正性\n"
            "2. 中等：相关性 AND (具体性 OR 矫正性)\n"
            "3. 无效：相关性 NOT (具体性 AND 矫正性)\n"
            "4. 无关：NOT 相关性"
        )
        
        说明标签 = ttk.Label(说明帧, text=说明文本, justify=tk.LEFT, foreground="green")
        说明标签.pack(padx=10, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = 麻醉反馈分析系统(root)
    root.mainloop()