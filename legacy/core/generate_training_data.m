% Define case name, input dimension, number of samples
case_name   = getenv('CASE_NAME');
input_dim   = str2double(getenv('INPUT_DIM'));
num_samples = str2double(getenv('NUM_SAMPLES'));

% Fallback to defaults if any input is missing or invalid
if isempty(case_name),   case_name = 'case300'; end
if isnan(input_dim),     input_dim = 60;       end
if isnan(num_samples),   num_samples = 100000;  end

fprintf('Running case %s with input dimension %d and %d samples...\n', case_name, input_dim, num_samples)

% Case-specific settings
clear settings
epB_struct = struct('case9', 100, 'case300', 10);
if isfield(epB_struct, case_name)
    settings.epB = epB_struct.(case_name);
else
    settingsepB = 10;
end
epL_struct = struct('case300', 80);
if isfield(epL_struct, case_name)
    settings.line_prob = 'all';
    settings.epL = epL_struct.(case_name);
else
    settings.epL = 0;
end

% Create MATPOWER case
model = eval(case_name);
num_buses = size(model.bus, 1);

% Get load buses
Pd_og = model.bus(:, 3);
Qd_og = model.bus(:, 4);
load_buses_idx = find(Pd_og ~= 0 | Qd_og ~= 0);
num_load_buses = size(load_buses_idx, 1);

% Track feasible inputs
num_feas = 0;

% Initialize array to store Pd, Qd, and costs
cost_data = [];

% Get sample points
input_filename = sprintf('../input_data/%s/input_%s_%dd_%dsamples.csv', case_name, case_name, input_dim, num_samples);
input_data = table2array(readtable(input_filename));
input_data = input_data(:, 2:end);
assert(all(size(input_data) == [num_samples, 2 * num_load_buses]));
Pd_arr = input_data(:, 1:num_load_buses);
Qd_arr = input_data(:, num_load_buses+1:end);

% File save information
% Headers Pd1 Pd3 ... Pd8 Qd1 Qd3 ... Qd8 Cost Feas_Flag Global_Opt
Pd_headers = strcat("Pd", string(load_buses_idx));
Qd_headers = strcat("Qd", string(load_buses_idx));
headers = [Pd_headers', Qd_headers', "Cost", "Feas_Flag", "Global_Opt"];
results_filename = sprintf('../training_data/%s/training_%s_%dd_%dsamples.csv', case_name, case_name, input_dim, num_samples);
temp_filename = sprintf('../training_data/temp/temp_%s_%dd_%dsamples.mat', case_name, input_dim, num_samples);
save(temp_filename, 'cost_data');
save_results = true;

tic
for i = 1:num_samples
       
    % Re-load model
    model = eval(case_name);
  
    % Set load at buses
    model.bus(load_buses_idx, 3) = Pd_arr(i, :)'; % Pseudo-random Pd
    model.bus(load_buses_idx, 4) = Qd_arr(i, :)'; % Pesudo-random Qd

    try
        % Solve model
        results = OPF_Solver(model, settings);
        
        % Extract Pd, Qd, and cost
        Pd = model.bus(load_buses_idx, 3)';  % Transpose to row vector
        Qd = model.bus(load_buses_idx, 4)';  % Transpose to row vector
        cost = results.sdp.cost;
    
        % Extract optimal recovered solution
        V = results.rec.V;
        Sg = results.rec.Sg;
        Sb = results.rec.Sb;
    
        % Assert SDP cost = recovered cost, recovered solution is feasible,
        % SDP solution is optimal --> recovered solution is global opt.
        global_opt = ( ...
            abs(results.sdp.cost - results.rec.cost) < 1e-1 & ...
            results.feas_flag & ...
            results.sdp.opt ...
        );
    
        fprintf('SDP Cost: %.2f | Rec. Cost: %.2f | Feas. Flag %.1f | Opt. Flag %.1f\n', results.sdp.cost, results.rec.cost, results.feas_flag, results.sdp.opt)
        if global_opt
            fprintf('Sample %d solved to global optimality\n', i)
        else
            fprintf('Sample %d NOT solved to global optimality\n', i)
        end

        % Append to data array
        cost_data = [cost_data; Pd, Qd, cost, results.feas_flag, global_opt];

        % Write out temporary file every 10 iterations
        if mod(i, 10) == 0
            save(temp_filename, 'cost_data', '-append');
        end

        % Track number of feasible data points
        num_feas = num_feas + results.feas_flag;

    catch ME
        % Handle the error and skip to the next iteration
        fprintf('Sample %d: Error occurred - %s\n', i, ME.message);
        continue; % Skip this sample and move to the next one
    end

end
toc

fprintf('Number of feasible solutions: %d\n', num_feas)

% Save data to CSV file
if save_results
    % Convert to table and write to CSV
    data_table = array2table(cost_data, 'VariableNames', headers);
    writetable(data_table, results_filename);
    
    fprintf('Results saved to %s\n', results_filename);
    
    % Clean up the temporary file after all computations are done
    delete(temp_filename);
    
    % Load temp data
    % load(temp_filename)
end