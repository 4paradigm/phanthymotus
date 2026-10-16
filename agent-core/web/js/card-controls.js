/** Optional schema metadata for readable fields and preview/confirm execution. */
export function fieldOptions(def) {
  const values = def.enum || (def.oneOf || []).map(o => o.const);
  const titles = new Map((def.oneOf || []).map(o => [o.const, o.title]));
  return values.map(value => ({ value, label: titles.get(value) || String(value) }));
}

export function confirmationText(rule, plan) {
  const lines = [rule.message || '确认执行本次操作？'];
  if (plan?.start_deg) lines.push(`当前角度（°）：${plan.start_deg.join('，')}`);
  if (plan?.target_deg) lines.push(`目标角度（°）：${plan.target_deg.join('，')}`);
  if (plan?.delta_deg) lines.push(`实际变化（°）：${plan.delta_deg.join('，')}`);
  if (plan?.boundary_adjustment_deg?.some(v => v !== 0)) {
    lines.push(`本次使能还会修正角度（°）：${plan.boundary_adjustment_deg.join('，')}`);
  }
  if (plan?.target_mm !== undefined) lines.push(`目标开度：${plan.target_mm} mm`);
  if (plan?.torque_nm !== undefined) lines.push(`力矩：${plan.torque_nm} N·m`);
  if (plan?.target_mode) lines.push(`目标模式：${plan.target_mode === 'can' ? '电脑控制（CAN）' : '待机'}`);
  return lines.join('\n\n');
}

export async function confirmedAction(schema, args, { call, confirm, showPreview = () => {} }) {
  const rule = schema?.['x-confirmation']?.[args.action];
  if (!rule) return call(args);
  let plan;
  if (rule.preview) {
    plan = await call({ ...args, ...rule.preview });
    showPreview(plan);
    if (plan.allowed === false) {
      throw new Error([plan.message || (plan.missing_conditions || []).join('；'), plan.next_step].filter(Boolean).join('\n'));
    }
    // Already at this mode/enable state: the driver deliberately sent no command.
    if (plan.confirmed === true && plan.executed === false) return plan;
    if (!plan.preview_id) throw new Error('没有取得有效预览；请重新读取状态后再操作');
  }
  if (!await confirm(confirmationText(rule, plan))) {
    return { executed: false, cancelled: true, message: '已取消本次执行；未发送执行指令' };
  }
  const execution = { ...args, ...rule.execute };
  if (plan?.preview_id) execution.preview_id = plan.preview_id;
  return call(execution);
}
