# 部署 Modal UX 改进方案

## 问题总结

经过全面审查UX 问题需要改进：

### 1. 进度反馈不一致 ⚠️ 高优先级
- **现状**：普通 driver 显示新进度窗口，Core 更新使用旧轮询
- **影响**：用户困惑，Core 更新（最重要）反而体验最差
- **解决**：统一使用 `DeployProgressUI`

### 2. 移动端进度窗口未优化 ⚠️ 高优先级  
- **现状**：固定 500px 宽度，在小屏幕溢出
- **影响**：移动端体验差，无法正常查看进度
- **解决**：响应式布局 + 底部抽屉样式

### 3. 确认 Modal 信息层级不清晰 🔸 中优先级
- **现状**：版本号太长、Channel 不突出、缺少影响说明
- **影响**：用户不清楚自己在做什么
- **解决**：优化信息层级、添加预估时间和影响说明

### 4. 顶栏更新提示缺少预检查 🔸 中优先级
- **现状**：点击"更新"直接开始，不检查磁盘/网络
- **影响**：Core 更新失败代价高
- **解决**：先显示预检查结果，再确认

---

## 改进方案

### 1️⃣ 统一进度反馈（所有入口）

#### 修改目标
让 Core 更新也使用 `DeployProgressUI`，保持体验一致。

#### 需要的后端支持
创建 `/api/system/update-v2` 端点，支持 WebSocket 进度推送。

#### 前端修改
**文件**: `agent-core/web/js/deploy-panel.js`
**位置**: 第 986-1001 行

---

### 2️⃣ 移动端进度窗口优化

#### 设计原则
- **Desktop**: 居中模态框（当前样式保持）
- **Mobile (<768px)**: 底部抽屉，可向下滑动关闭

#### CSS 修改
**文件**: `agent-core/web/js/deploy-progress.js`

增加响应式样式：

\`\`\`javascript
// 在 style.textContent 中添加 @media 查询
@media (max-width: 768px) {
    .deploy-progress-modal {
        align-items: flex-end;
    }
    .deploy-progress-content {
        min-width: unset;
        width: 100%;
        max-width: 100%;
        max-height: 70vh;
        border-radius: 16px 16px 0 0;
        animation: slideUpMobile 0.3s ease-out;
    }
    .deploy-progress-header {
        padding-top: 8px;
    }
    .deploy-progress-header::before {
        content: '';
        display: block;
        width: 40px;
        height: 4px;
        background: #666;
        border-radius: 2px;
        margin: 0 auto 12px;
    }
}

@keyframes slideUpMobile {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
}
\`\`\`

---

### 3️⃣ 确认 Modal 信息优化

#### 当前问题
\`\`\`html
<!-- 现在：信息扁平，层级不清 -->
<div class="deploy-confirm-item-name">Perception 感知层</div>
<div class="deploy-confirm-item-versions">
  <span>release.260901.abc</span> → <span>release.260909.2a74517</span>
</div>
\`\`\`

#### 改进设计
\`\`\`html
<!-- 改进：层级清晰，突出关键信息 -->
<div class="deploy-confirm-item">
  <div class="deploy-confirm-item-header">
    <span class="deploy-confirm-item-name">Perception 感知层</span>
    <span class="deploy-confirm-channel-badge release">Release</span>
  </div>
  
  <div class="deploy-confirm-versions">
    <div class="deploy-confirm-version-block">
      <div class="deploy-confirm-version-label">当前版本</div>
      <div class="deploy-confirm-version-tag">260901.abc</div>
    </div>
    
    <div class="deploy-confirm-arrow">→</div>
    
    <div class="deploy-confirm-version-block highlight">
      <div class="deploy-confirm-version-label">新版本</div>
      <div class="deploy-confirm-version-tag">260909.2a74517</div>
    </div>
  </div>
  
  <div class="deploy-confirm-meta">
    <span class="deploy-confirm-meta-item">
      <svg>⏱</svg> 预计 5-8 分钟
    </span>
    <span class="deploy-confirm-meta-item">
      <svg>⚠️</svg> 需要重启服务
    </span>
  </div>
</div>
\`\`\`

#### 视觉改进
- **版本号简化**：只显示日期+短哈希（`260909.2a74517` 而非完整镜像路径）
- **Channel 徽章**：醒目的颜色标签（Stable=绿色，Release=黄色，Preview=橙色）
- **元信息**：预估时间、影响范围、镜像大小

---

### 4️⃣ 顶栏更新流程优化

#### 当前流程
\`\`\`
用户点击"更新" → 直接开始部署 → 可能失败（磁盘不足）
\`\`\`

#### 优化流程
\`\`\`
用户点击"更新" 
  ↓
显示预检查 Modal
  ✓ 磁盘空间：45 GB 可用
  ✗ 网络连接：较慢（120ms）
  ✓ 仓库认证：已连接
  
  [建议稍后再试] [继续更新]
  ↓
用户确认 → 显示进度窗口 → 完成
\`\`\`

#### 实现
**文件**: `agent-core/web/js/app.js` 或相关文件

修改顶栏"更新"按钮的点击处理：

\`\`\`javascript
document.getElementById('btn-update').addEventListener('click', async () => {
  const updateData = /* ... 从 update banner 获取版本信息 ... */;
  
  // 1. 先显示预检查
  const preflight = await fetch('/api/drivers/agent-core/preflight');
  const result = await preflight.json();
  
  // 2. 显示预检查结果 Modal
  showPreflightModal(result.data, () => {
    // 3. 用户确认后，显示进度并开始部署
    showDeployConfirmModal([{
      label: 'Agent Core',
      currentTag: currentVersion,
      newTag: updateData.version,
      channel: 'release'
    }], () => {
      _executeCoreDeploy(updateData.image);
    });
  });
});
\`\`\`

---

## 设计语言一致性

### 当前设计系统
- **颜色**：暖色调 parchment (#F5F0E8) + 赤陶色强调 (#C4673A)
- **字体**：Plus Jakarta Sans (UI) + JetBrains Mono (代码)
- **圆角**：10px (卡片) / 6px (按钮) / 4px (标签)
- **阴影**：轻柔 (0 4px 16px rgba(0,0,0,.10))

### 部署 Modal 应遵循的原则

1. **渐进式披露**
   - 不要一次展示所有信息
   - 预检查 → 确认 → 进度 → 结果（分步展示）

2. **清晰的视觉层级**
   - 主要操作（升级/安装）使用强调色
   - 次要操作（取消/查看日志）使用灰色
   - 危险操作（卸载/停止）使用红色

3. **反馈即时可见**
   - 按钮状态（loading/disabled/success）
   - 进度百分比 + 速度 + ETA
   - 错误提示 + 解决建议

4. **移动优先**
   - 触摸目标 ≥ 44px
   - 底部抽屉优于居中 modal
   - 大拇指可达区域放主要操作

---

## 实现优先级

### Phase 1: 核心体验统一（1-2天）
- ✅ 已完成：普通 driver 使用 DeployProgressUI
- ⬜ TODO: Core 更新使用相同 UI
- ⬜ TODO: 移动端响应式优化

### Phase 2: 信息优化（1天）
- ⬜ 确认 Modal 信息层级重构
- ⬜ 添加预估时间和影响说明
- ⬜ Channel 徽章醒目化

### Phase 3: 预检查集成（0.5天）
- ⬜ 顶栏更新加预检查
- ⬜ 预检查结果 Modal 设计

---

## 测试检查清单

### Desktop
- [ ] 从"我的服务"升级看到进度窗口
- [ ] 从"驱动市场"安装看到进度窗口
- [ ] 从顶栏更新 Core 看到进度窗口
- [ ] 版本切换下拉菜单正常工作
- [ ] 确认 Modal 信息清晰可读
- [ ] 进度条平滑更新
- [ ] 错误提示清晰且有建议

### Mobile (<768px)
- [ ] 进度窗口从底部滑入
- [ ] 可以向下滑动关闭
- [ ] 版本选择使用底部抽屉
- [ ] 所有按钮触摸目标足够大
- [ ] 文字在小屏幕不会溢出
- [ ] 横屏也能正常显示

### 边缘情况
- [ ] 磁盘空间不足的预检查警告
- [ ] 网络断开时的错误处理
- [ ] 同时部署多个 driver
- [ ] 部署中刷新页面
- [ ] 非常长的版本号

---

## 参考设计

### 类似产品的最佳实践

1. **Docker Desktop** - 镜像拉取进度
   - 逐层显示进度
   - 总体百分比 + 速度
   - 失败时显示具体层的错误

2. **npm/yarn** - 依赖安装进度
   - 清晰的阶段划分（resolve → fetch → link）
   - 简洁的进度反馈
   - 时间预估

3. **Vercel** - 部署进度
   - 实时日志流
   - 每个步骤的耗时
   - 成功/失败的视觉标识

