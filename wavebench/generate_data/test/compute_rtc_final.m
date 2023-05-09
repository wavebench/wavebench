function final_pressure = compute_rtc_final( ...
  medium_sound_speed, medium_density, domain_dx, initial_pressure)
  % Compute the final pressure field using MATLAB's k-Wave toolbox.

  sizes = size(medium_sound_speed);
  sidelen = sizes(1);
  kgrid = kWaveGrid( ...
      sidelen, domain_dx, sidelen, domain_dx);
  medium.sound_speed = medium_sound_speed;  % [m/s]
  medium.density = medium_density;

  kgrid.makeTime(medium.sound_speed, 0.3, 0.2);
  % kgrid.setTime(kgrid.Nt + 1, kgrid.dt);
  kgrid.setTime(kgrid.Nt, kgrid.dt);


  source.p0 = initial_pressure;
  sensor.record = {'p_final'};

  % run the simulation
  sensor_data = kspaceFirstOrder2D( ...
    kgrid, medium, source, sensor, ...
    'PMLInside', true, ...
    'RecordMovie', false, ...
    'PlotSim', false, ...
    'Smooth', true);

      % 'PMLInside', false, 'RecordMovie', false);
  final_pressure = sensor_data.p_final;
end