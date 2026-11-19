/** Presentation-only mapping to Unitree-style fields; no custom panel or CSS. */
export function isBatteryStatus(value) {
  return value && Array.isArray(value.batteries) && value.expected_battery_count === 2
    && value.units?.soc_pct === '%' && typeof value.state === 'string';
}
const finite = value => typeof value === 'number' && Number.isFinite(value);
const numeric = value => finite(value) ? value : null;
const millivolts = value => finite(value) ? Math.round(value * 1000) : null;
const units = {
  soc: '%', voltage: 'mV',
  cell_vol_min: 'mV', cell_vol_max: 'mV', temperature: '°C',
  cycle: 'count', remaining_capacity: 'mAh', full_charge_capacity: 'mAh',
};

export function batteryEntries(data, elapsed = 0) {
  const fresh = finite(data.age_ms) && data.age_ms + Math.max(0, elapsed) <= 3000 && elapsed <= 3000;
  return ['1', '2'].map(id => {
    const pack = data.batteries.find(p => p?.id === id);
    const valid = fresh && data.available === true && pack?.available === true;
    const soc = valid && finite(pack.soc_pct) && pack.soc_pct >= 0 && pack.soc_pct <= 100 ? pack.soc_pct : null;
    return {
      values: {
        battery_id: id,
        soc,
        voltage: valid ? millivolts(pack.voltage_v) : null,
        cell_vol_min: valid ? millivolts(pack.cell_voltage_min_v) : null,
        cell_vol_max: valid ? millivolts(pack.cell_voltage_max_v) : null,
        // Match the vendor display: highest sensor temperature in this pack's latest frame.
        temperature: valid ? numeric(pack.temperature_max_c) : null,
        cycle: valid ? numeric(pack.cycle_count) : null,
      },
      details: {
        temperature_source: 'max_temperature (highest pack sensor reading)',
        remaining_capacity: valid ? numeric(pack.remaining_capacity_mah) : null,
        full_charge_capacity: valid ? numeric(pack.full_charge_capacity_mah) : null,
        bms_hardware_version: valid ? pack.bms_hardware_version ?? null : null,
        bms_software_version: valid ? pack.bms_software_version ?? null : null,
        state: !fresh && data.available ? 'stale' : valid ? 'online' : pack?.state || data.state,
        available: valid,
        age_ms: finite(data.age_ms) ? Math.round(data.age_ms + Math.max(0, elapsed)) : null,
        received_at_ms: data.received_at_ms ?? null,
        timestamp_source: data.timestamp_source ?? null,
        source_timestamp: data.source_timestamp ?? null,
        units,
      },
    };
  });
}

export function batteryLatestValues(data, elapsed = 0) {
  const result = {};
  for (const {values, details} of batteryEntries(data, elapsed)) {
    const prefix = `battery_${values.battery_id}.`;
    for (const [key, value] of Object.entries({...values, ...details})) {
      if (key !== 'battery_id' && key !== 'units') result[prefix + key] = value;
    }
  }
  for (const [key, value] of Object.entries(units)) result[`units.${key}`] = value;
  return result;
}
