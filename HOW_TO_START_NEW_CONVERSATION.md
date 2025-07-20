# 🚀 新对话开始指南

> **为什么需要这个指南？**  
> AI对话是无状态的，每次新对话都需要重新建立项目上下文。这个指南帮助您快速让新的AI理解项目状态。

## 📋 标准开场白模板

### 方案1：完整上下文（推荐）
```
我需要继续F6Transform项目的开发工作。这是一个相机标定和3D测量的Python项目。

请按顺序阅读以下文件来了解项目状态：
1. PROJECT_STATUS.md - 项目整体状态和进度
2. .claude/context.md - Claude Code专用上下文
3. calibration/docs/CLAUDE.md - 技术背景详细说明

阅读完成后，请告诉我：
- 您理解的项目核心技术是什么？
- 当前的项目状态如何？
- 下一步主要任务是什么？
```

### 方案2：快速启动
```
我需要继续F6Transform相机标定项目。请先读取PROJECT_STATUS.md了解当前状态，然后告诉我您对项目的理解程度。
```

### 方案3：特定任务导向
```
我需要在F6Transform项目中[具体任务描述]。
请先阅读PROJECT_STATUS.md和.claude/context.md了解项目背景，然后我们开始工作。
```

## ✅ 验证AI理解程度的问题

### 核心技术理解
- [ ] AI能说出项目的核心创新点（单参数内方位、自标定算法等）
- [ ] AI理解项目的应用场景（快速运动目标3D测量）
- [ ] AI知道主要算法模块的作用

### 项目状态理解  
- [ ] AI知道当前完成了哪些工作
- [ ] AI了解最近的重构工作（目录整理）
- [ ] AI清楚下一步的主要任务

### 代码结构理解
- [ ] AI知道新的目录结构布局
- [ ] AI了解如何运行测试和工具
- [ ] AI知道各个模块的位置和作用

## 🔧 常用快速指令

### 查看项目状态
```bash
# 查看最近提交
git log --oneline -10

# 查看当前目录结构  
tree calibration/ -L 2

# 运行测试验证
cd calibration/tests && python3 backPrjctTest.py
```

### 重新生成数据
```bash
cd calibration/tools
python3 saveParam.py
python3 save_mapping_table.py
python3 verify_points.py
```

## 📚 关键文档快速索引

| 文档 | 作用 | 何时查阅 |
|------|------|----------|
| `PROJECT_STATUS.md` | 项目整体状态 | 每次新对话开始 |
| `.claude/context.md` | Claude专用上下文 | Claude Code使用时 |
| `calibration/docs/CLAUDE.md` | 技术详细背景 | 需要了解算法细节时 |
| `calibration/docs/README.md` | 算法说明 | 需要算法实现细节时 |
| `requirements.md` | 项目需求 | 需要了解项目目标时 |

## ⚠️ 常见问题和解决方案

### Q: AI说找不到某个文件
**A**: 检查是否使用了正确的相对路径，目录已重构

### Q: AI不理解项目的技术背景  
**A**: 让AI先读 `calibration/docs/CLAUDE.md`

### Q: AI不知道当前进度
**A**: 让AI读 `PROJECT_STATUS.md` 和查看 `git log`

### Q: AI对目录结构困惑
**A**: 运行 `tree calibration/` 显示最新结构

## 🎯 不同场景的开场白

### 场景1：继续开发工作
```
继续F6Transform项目开发。请读PROJECT_STATUS.md了解当前状态，然后我们继续[具体任务]。
```

### 场景2：修复问题
```
F6Transform项目遇到了[问题描述]。请先了解项目背景（读PROJECT_STATUS.md），然后帮我分析问题。
```

### 场景3：添加新功能
```
需要为F6Transform项目添加[新功能]。请先阅读项目状态和技术文档，然后我们讨论实现方案。
```

### 场景4：代码重构
```
需要对F6Transform项目进行[重构任务]。请先熟悉当前代码结构（看PROJECT_STATUS.md），然后开始工作。
```

## 📝 更新提醒

**每次重要进展后，请更新：**
1. `PROJECT_STATUS.md` - 更新项目状态和完成工作
2. `.claude/context.md` - 更新当前焦点和下一步任务  
3. `calibration/docs/CLAUDE.md` - 更新技术状态
4. Git提交信息 - 详细记录变更内容

**让新对话快速上手的关键：**
- 文档要实时更新
- 状态要准确反映当前情况
- 下一步任务要明确具体

---

*保持这个指南的更新，让每次新对话都能快速建立有效上下文！*