/** Legacy import path. QQ onboarding no longer has administrator pairing.
 * The production entry uses qq-onboarding-v2.js directly; keep this forwarding
 * module so older extensions cannot accidentally reopen retired UI.
 */
export { openQQWizard } from './qq-onboarding-v2.js';
