import os

for medium_type in ['gaussian_lens', 'grf_anisotropic', 'grf_isotropic']:
  for initial_pressure_type in ['thick_lines', 'mnist']:
    os.system(f'python generate_data_rtc.py --device_id 0 --initial_pressure_type {initial_pressure_type} --medium_type {medium_type}')
