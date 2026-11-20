/** Original test geometry. Dimensions, axes and origins are synthetic, not factory CAD. */
export function syntheticHeliosModel() {
  const links = new Set(['base_link']);
  const joints = [];
  const add = (name, type, parent, child, xyz = '0 0 0.12', rpy = '0.1 0.2 0.3') => {
    links.add(child);
    joints.push(`<joint name="${name}" type="${type}"><parent link="${parent}"/>`
      + `<child link="${child}"/><origin xyz="${xyz}" rpy="${rpy}"/>`
      + '<axis xyz="0 0 1"/></joint>');
  };
  for (let i = 1; i <= 4; i++) {
    add(`body_joint_${i}`, 'revolute', i === 1 ? 'base_link' : `body_link_${i-1}`, `body_link_${i}`);
  }
  for (const [side, sign] of [['left', 1], ['right', -1]]) {
    add(`${side}_fixed`, 'fixed', 'body_link_4', `${side}_base`, `0 ${sign * 0.18} 0.08`);
    for (let i = 1; i <= 7; i++) {
      add(`${side}_joint_${i}`, 'revolute', i === 1 ? `${side}_base` : `${side}_link_${i-1}`,
        `${side}_link_${i}`, `0.08 ${sign * 0.03} 0.04`);
    }
    add(`${side}_tip`, 'fixed', `${side}_link_7`, `${side}_flan_link`);
  }
  add('head_joint_1', 'revolute', 'body_link_4', 'head_link_1');
  add('head_joint_2', 'revolute', 'head_link_1', 'head_link_2');
  for (const sensor of ['ChestRGBD', 'HeadRGBD', 'RGB1', 'RGB2', 'RGB3']) {
    add(`test_${sensor}`, 'fixed', sensor === 'ChestRGBD' ? 'body_link_4' : 'head_link_2', `${sensor}_link`);
  }
  for (let i = 1; i <= 4; i++) add(`test_wheel_${i}`, 'fixed', 'base_link', `test_wheel_link_${i}`);
  return '<robot name="Rokae_Robot_2.1_AR5_v1.1">'
    + [...links].map(name => `<link name="${name}"/>`).join('') + joints.join('') + '</robot>';
}
