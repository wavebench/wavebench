function measurements = compute_is_measurements( ...
  medium_sound_speed, medium_density, domain_dx, initial_pressure, pml_size)
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
  sensor.mask = zeros(sidelen, sidelen);

  sensor.mask(pml_size, :) = 1.0;
  sensor.record = {'p', 'p_final'};

  % run the simulation
  sensor_data = kspaceFirstOrder2D( ...
    kgrid, medium, source, sensor, ...
    'PMLInside', true, ...
    'RecordMovie', false, ...
    'PlotSim', false, ...
    'PMLSize', pml_size, ...
    'Smooth', true);

      % 'PMLInside', false, 'RecordMovie', false);
  measurements = sensor_data.p;
end