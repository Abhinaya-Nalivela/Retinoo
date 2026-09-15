% MATLAB Toolbox Verification Checklist for SIH 26038
required_toolboxes = {
    'Image Processing Toolbox',
    'Computer Vision Toolbox',
    'Deep Learning Toolbox',
    'Statistics and Machine Learning Toolbox',
    'Simulink'
};

disp('SIH 26038: Toolbox Verification Checklist');
disp('------------------------------------------');
v = ver;
installed = {v.Name};

for i = 1:length(required_toolboxes)
    t = required_toolboxes{i};
    if any(strcmp(installed, t))
        disp(['[PASS] ', t, ' is installed.']);
    else
        disp(['[OPTIONAL/MISSING] ', t, ' not detected. Python fallback engine active.']);
    end
end
