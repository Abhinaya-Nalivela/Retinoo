% MATLAB Environment Diagnostic for SIH 26038
disp('===========================================================');
disp('SIH 26038: MATLAB Environment Diagnostic');
disp('===========================================================');
disp(['MATLAB Version: ', version]);
disp(['Release: ', version('-release')]);
disp(['Architecture: ', computer]);

v = ver;
installed = {v.Name};
disp(['Total Installed Toolboxes: ', num2str(length(installed))]);
disp('===========================================================');
