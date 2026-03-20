clear all

% Folder with saved iterations
folder = 'stored_arrays';  %'saved_arrays';
files = dir(fullfile(folder, 'iteration_*.mat'));  % or .npz converted to .mat
nFiles = length(files);

% Start iteration
currentIdx = 1;
% inner_model_pcd = pcread('full_outer_inner_smoothed_part_only.ply');
inner_model_pcd = pcread('full_outer_inner_part_only.ply');
inner_model_pcd = rectify_model(inner_model_pcd);
outer_model_pcd = pcread('full_outer_outer_part_only.ply');
outer_model_pcd = rectify_model(outer_model_pcd);

% plot_radial_displacement(inner_model_pcd)
% rectify_tyre_center_and_plot(pcdownsample(inner_model_pcd,"gridAverage",0.01))
% compare_radial(pcdownsample(inner_model_pcd,"gridAverage",0.01),pcdownsample(outer_model_pcd,"gridAverage",0.01))

% for i = 1:10
%     phi = (i * 36)*(pi/180);
%     R2 = [1, 0, 0;
%            0, cos(phi), -sin(phi);
%            0, sin(phi), cos(phi)];
%     p_inner = (R2 * inner_model_pcd.Location')';
%     p_outer = (R2 * outer_model_pcd.Location')';
%     inner_model_pcd = pointCloud(p_inner);
%     outer_model_pcd = pointCloud(p_outer);
%     [theta_grid, width_grid, R_diff,Ro] = compare_radial(inner_model_pcd,outer_model_pcd);
% 
%     pc = radial_map_to_pointcloud(theta_grid, width_grid, R_diff,Ro);
% 
%     pcshow(pc)
% end

[theta_grid, width_grid, R_diff,Ro] = compare_radial(inner_model_pcd,outer_model_pcd);

pc = radial_map_to_pointcloud(theta_grid, width_grid, R_diff,Ro);

pcshow(pc)

function pc = radial_map_to_pointcloud(theta_grid, width_grid, R_diff, Ro)
% Create meshgrid
[Theta, Width] = meshgrid(theta_grid, width_grid);  % size: nWidth x nTheta
R = R_diff';  % transpose to match width vs theta
Ro = Ro';

% Flatten
Theta = Theta(:);
Width = Width(:);
R = R(:);
Ro = Ro(:);

% Remove NaNs
valid = ~isnan(R);
Theta = Theta(valid);
Width = Width(valid);
R = R(valid);
Ro = Ro(valid);

% Convert to Cartesian (Y,Z radial plane, X = width)
Y = Ro .* cos(Theta);
Z = Ro .* sin(Theta);
X = Width;

% Combine into Nx3
points = [X(:), Y(:), Z(:)];

% Normalize colors to R range
% cdata = (R - 0.04) / (0.05 - 0.04);  % 0-1
R_clipped = min(max(R,0.035),0.05);
cdata = rescale(R_clipped, 0, 1); 
cdata = cdata(:);
% Create point cloud with color
cmap = jet(256);
idx = round(cdata * 255) + 1;
idx = max(min(idx,256),1);
rgb = cmap(idx, :);   % Nx3

% Create point cloud
pc = pointCloud(points, 'Color', rgb);

% Visualize
figure;
pcshow(pc, 'MarkerSize', 50);
xlabel('Width'); ylabel('Y'); zlabel('Z');
title('Radial Displacement Colored Point Cloud');
colormap jet; colorbar;
end


function [theta_grid, width_grid, R_diff,Ro] = compare_radial(inner_pcd, outer_pcd)
    if isa(inner_pcd, 'pointCloud')
        Pi = double(inner_pcd.Location) + [0,0,0];
    else
        Pi = double(inner_pcd);
    end
    if isa(outer_pcd, 'pointCloud')
        Po = double(outer_pcd.Location) + [0,0,0];
    else
        Po = double(outer_pcd);
    end
    
    % Assuming:
    %   width along X, radial plane YZ
    xi = Pi(:,2); yi = Pi(:,3);  width_i = Pi(:,1);
    xo = Po(:,2); yo = Po(:,3);  width_o = Po(:,1);
    
    r_i = sqrt(xi.^2 + yi.^2);
    r_o = sqrt(xo.^2 + yo.^2);
    
    theta_i = atan2(yi, xi);
    theta_o = atan2(yo, xo);
    
    nTheta = 360;   % 1-degree bins
    nWidth = 360;   % number of width bins
    
    theta_edges = linspace(-pi, pi, nTheta+1);
    width_edges = linspace(min([width_i; width_o]), max([width_i; width_o]), nWidth+1);
    
    theta_grid = (theta_edges(1:end-1) + theta_edges(2:end))/2;
    width_grid = (width_edges(1:end-1) + width_edges(2:end))/2;
    
    theta_bin_i = discretize(theta_i, theta_edges);
    width_bin_i = discretize(width_i, width_edges);
    valid_i = ~isnan(theta_bin_i) & ~isnan(width_bin_i);
    theta_bin_i = theta_bin_i(valid_i);
    width_bin_i = width_bin_i(valid_i);
    r_i = r_i(valid_i);
    
    Ri = accumarray([theta_bin_i, width_bin_i], r_i, [nTheta, nWidth], @mean, NaN);
    
    theta_bin_o = discretize(theta_o, theta_edges);
    width_bin_o = discretize(width_o, width_edges);
    valid_o = ~isnan(theta_bin_o) & ~isnan(width_bin_o);
    theta_bin_o = theta_bin_o(valid_o);
    width_bin_o = width_bin_o(valid_o);
    r_o = r_o(valid_o);
    
    Ro = accumarray([theta_bin_o, width_bin_o], r_o, [nTheta, nWidth], @mean, NaN);
    
    R_diff = Ro - Ri;
    
    figure;
    imagesc(rad2deg(theta_grid), width_grid, R_diff');  % transpose so width is y-axis
    set(gca,'YDir','normal');
    colormap jet; colorbar;
    xlabel('Theta (deg)');
    ylabel('Tyre width');
    title('Outer − Inner Radial Displacement (m)');
    clim([0.035,0.05])
    
end

function rectify_tyre_center_and_plot(model_pcd)
% RECTIFY_TYRE_CENTER_AND_PLOT
% Takes Nx3 or pointCloud. Assumes:
%  - x = pts(:,1)
%  - y = pts(:,2)   % (you previously used y as tyre width; keep same convention)
%  - z = pts(:,3)
% Radial computed in X-Y plane (r = sqrt(x^2 + y^2)), theta = atan2(y,x).
%
% Outputs before/after plots and prints centre estimates and radial stats.

    % --- load points ---
    if isa(model_pcd,'pointCloud')
        pts = model_pcd.Location;
        pts = pts(abs(pts(:,1)) < 0.055,:);
    else
        pts = model_pcd;
    end
    if size(pts,2) ~= 3
        error('Input must be Nx3 or a pointCloud');
    end
    x = pts(:,1);
    y = pts(:,2);
    z = pts(:,3);

    % --- compute theta and radial (current origin assumed at 0,0) ---
    theta = atan2(y, z);            % radians (-pi..pi)
    theta_deg = rad2deg(theta);
    r = sqrt(y.^2 + z.^2);

    % --- Fit first harmonic r(θ) = R0 + a*cosθ + b*sinθ ---
    A = [ones(size(theta)), cos(theta), sin(theta)];   % design
    coeff = A \ r;   % least squares solution
    R0 = coeff(1);
    a = coeff(2);
    b = coeff(3);

    % Centre estimate from harmonic fit (approx)
    centre_harm = [a, b];   % (x_offset, y_offset)
    
    % --- Algebraic circle fit (linear least squares) for cross-check ---
    % Solve for D = [A B C]' in x^2 + y^2 + A x + B y + C = 0
    M = [y, z, ones(size(y))];
    rhs = -(y.^2 + z.^2);
    D = M \ rhs;
    Acoef = D(1); Bcoef = D(2); Ccoef = D(3);
    centre_alg = [-Acoef/2, -Bcoef/2];
    R_alg = sqrt((Acoef^2 + Bcoef^2)/4 - Ccoef);

    % --- Report raw stats ---
    fprintf('Harmonic-fit centre estimate: x = %.4f m, y = %.4f m\n', centre_harm(1), centre_harm(2));
    fprintf('Algebraic circle centre estimate: x = %.4f m, y = %.4f m, R = %.4f m\n', centre_alg(1), centre_alg(2), R_alg);

    fprintf('\nRadial stats BEFORE recentering (m):\n');
    fprintf('  mean = %.6f, std = %.6f, min = %.6f, max = %.6f\n', mean(r), std(r), min(r), max(r));

    % --- Recenter using harmonic estimate (you can switch to centre_alg if you prefer) ---
    centre_to_use = centre_harm;   % choose harmonic fit by default
    % If you want algebraic: centre_to_use = centre_alg;
  
    y_corr = y - centre_to_use(1);
    z_corr = z - centre_to_use(2);
    r_corr = sqrt(y_corr.^2 + z_corr.^2);
    theta_corr = atan2(z_corr, y_corr);
    theta_corr_deg = rad2deg(theta_corr);

    fprintf('\nRadial stats AFTER recentering (m):\n');
    fprintf('  mean = %.6f, std = %.6f, min = %.6f, max = %.6f\n', mean(r_corr), std(r_corr), min(r_corr), max(r_corr));
    fprintf('  radial variation (peak-to-peak) BEFORE = %.4f mm, AFTER = %.4f mm\n', (max(r)-min(r))*1000, (max(r_corr)-min(r_corr))*1000);

    % --- Optional: per-width-slice centres (to detect tilt or axis misalignment) ---
    % Bin along width (y coordinate original) to estimate centre variation along width
    nbins = 30;
    edges = linspace(min(x), max(x), nbins+1);
    bin_centres = zeros(nbins,2);
    bin_counts = zeros(nbins,1);
    for i=1:nbins
        inbin = x >= edges(i) & x < edges(i+1);
        if sum(inbin) < 20
            bin_centres(i,:) = [NaN NaN];
            bin_counts(i) = 0;
            continue;
        end
        yi = y(inbin);
        zi = z(inbin);
        ri = sqrt(yi.^2 + zi.^2);
        Ai = [ones(size(ri)), cos(atan2(zi,yi)), sin(atan2(zi,yi))];
        ci = Ai \ ri;
        bin_centres(i,:) = [ci(2), ci(3)];  % harmonic center estimate per slice
        bin_counts(i) = sum(inbin);
    end

    % --- PLOTS: before and after radial map scatter (theta on x, width on y, color radial) ---
    figure('Name','Radial map BEFORE & AFTER recentering','Position',[100 100 1200 500]);

    subplot(1,2,1);
    scatter(rad2deg(theta), x, 6, r, 'filled');
    xlabel('Angle θ (deg)'); ylabel('Tyre width coordinate (y)');
    title('Before recentering (color = radial distance)'); colorbar; colormap jet;
    caxis([0.345 0.355]); % keep same color-limits you used earlier; adapt if needed
    xlim([-180 180]); grid on;

    subplot(1,2,2);
    scatter(rad2deg(theta_corr), x, 6, r_corr, 'filled'); 
    % note: we plot width in the same original frame for visual comparison (shift back)
    xlabel('Angle θ (deg)'); ylabel('Tyre width coordinate (y)');
    title('After recentering'); colorbar; colormap jet;
    caxis([0.345 0.355]);
    xlim([-180 180]); grid on;

    % --- centre variation along width plot ---
    figure('Name','Centre variation along width');
    xx = (edges(1:end-1)+edges(2:end))/2;
    plot(xx, bin_centres(:,1),'o-', 'DisplayName','x centre (harmonic)'); hold on;
    plot(xx, bin_centres(:,2),'s-', 'DisplayName','y centre (harmonic)');
    xlabel('Width coordinate (y)'); ylabel('Centre estimate (m)');
    title('Centre offset estimates vs tyre width (per-slice)');
    legend show; grid on;

    % --- show numeric summary of centre variation if available ---
    valid = ~isnan(bin_centres(:,1));
    if any(valid)
        fprintf('\nPer-slice centre variation (harmonic estimates):\n');
        fprintf('  x centre: mean = %.4f mm, std = %.4f mm, range = %.4f mm\n', mean(bin_centres(valid,1))*1000, std(bin_centres(valid,1))*1000, (max(bin_centres(valid,1))-min(bin_centres(valid,1)))*1000);
        fprintf('  y centre: mean = %.4f mm, std = %.4f mm, range = %.4f mm\n', mean(bin_centres(valid,2))*1000, std(bin_centres(valid,2))*1000, (max(bin_centres(valid,2))-min(bin_centres(valid,2)))*1000);
    end

    % --- final: apply correction to point cloud variable for downstream use ---
    pts_corrected = pts;
    pts_corrected(:,2) = y_corr;
    pts_corrected(:,3) = z_corr;
    % If you want to return the corrected point cloud, save it to workspace:
    assignin('base','pts_corrected',pts_corrected);
    fprintf('\nCorrected point cloud saved to workspace variable: pts_corrected\n');
    fprintf('Centre used for recentering: [%.6f, %.6f] m\n', centre_to_use(1), centre_to_use(2));
end

%%

chooseIdx = [3,4,5,6];%[1,170:5:210]; %150:5:180
% chooseIdx = 110:1:210;
data_iter = cell(numel(chooseIdx), 1);
for k = 1:numel(chooseIdx)
    idx = chooseIdx(k);
    data_iter{k} = load_data(idx,files);
end

% Load first iteration
data = load(fullfile(files(currentIdx).folder, files(currentIdx).name));
dt = data.t;
time = [0 cumsum(dt)];
inner_def = data.pcd_inner_deform;
inner_undef = data.pcd_inner_undeform;
outer_def = data.pcd_outer_deform;
outer_undef = data.pcd_outer_undeform;
tread_def = data.pcd_tread_deform;
tread_undef = data.pcd_tread_undeform;
contact_patch = data.pcd_contact_patch;
dist_inner_def = data.d_inner_deform;
dist_inner_to_outer_undef = data.d_inner_to_outer_undeform;
dist_inner_to_tread_undef = data.d_inner_to_tread_undeform;
curr_valid_mask = data.curr_valid_mask;
prev_valid_mask = data.prev_valid_mask;
contact_patch_mask = data.contact_patch_mask;

mask_inner_def = inner_def(:,3) > 0.1;
pcd_inner_def = pointCloud(inner_def(mask_inner_def,:));
pcd_inner_undef = pointCloud(inner_undef);
pcd_outer_def = pointCloud(outer_def(mask_inner_def,:));
pcd_outer_undef = pointCloud(outer_undef);
pcd_tread_def = pointCloud(tread_def);
pcd_tread_undef = pointCloud(tread_undef);
pcd_contact_patch = pointCloud(contact_patch);

pcd_inner_def = rotate_cam_view(pcd_inner_def);
pcd_inner_undef = rotate_cam_view(pcd_inner_undef);
pcd_outer_def = rotate_cam_view(pcd_outer_def);
pcd_outer_undef = rotate_cam_view(pcd_outer_undef);
pcd_tread_def = rotate_cam_view(pcd_tread_def);
pcd_tread_undef = rotate_cam_view(pcd_tread_undef);
pcd_contact_patch = rotate_cam_view(pcd_contact_patch);
%rigid transform and then take slices on theta;

colorPointCloud(pcd_inner_def, dist_inner_def(mask_inner_def), 'plasma')
% for i = 1:length(chooseIdx)
%     % showPC_Colorbar_Hist(pcd_inner_def,dist_inner_def(mask_inner_def),'plasma',chooseIdx, inner_model_pcd, files)
%     showPC_Colorbar_Hist(pcd_outer_def,dist_inner_def(mask_inner_def),'plasma',chooseIdx(i), inner_model_pcd, files(chooseIdx))
%     % showPC_Colorbar_Hist(pcd_tread_def,dist_inner_def(mask_inner_def),'plasma',chooseIdx, inner_model_pcd, files)
% end



% figure; hold on;
% pcshow(pcd_inner_def);
% % pcshow(pcd_outer_def);
% % pcshow(pcd_tread_def);
% hold off;
% legend('Inner','Outer','Tread');
% title('Multiple Point Clouds');


% create_3dscatter(inner,outer,currentIdx,files)
% plot_crossSection(inner_def,outer_def,currentIdx,files)

% showPC_Colorbar_Hist(pcd_inner_def,dist_inner_def(mask_inner_def),'plasma',currentIdx, inner_model_pcd, files)
% plot_longitudinal_circumferencial_angle_vs_deformation(data_iter,'plasma',chooseIdx, inner_model_pcd, files)
plot_radial_crossSection(data_iter,'plasma',chooseIdx, inner_model_pcd, files)

% track_rolling_points(data_iter,'plasma',chooseIdx, inner_model_pcd, files)


function input_data = load_data(currentIdx,files)
    data = load(fullfile(files(currentIdx).folder, files(currentIdx).name));
    dt = data.t;
    time = [0 cumsum(dt)];
    inner_def = data.pcd_inner_deform;
    inner_undef = data.pcd_inner_undeform;
    outer_def = data.pcd_outer_deform;
    outer_undef = data.pcd_outer_undeform;
    tread_def = data.pcd_tread_deform;
    tread_undef = data.pcd_tread_undeform;
    contact_patch = data.pcd_contact_patch;
    dist_inner_def = data.d_inner_deform;
    dist_inner_to_outer_undef = data.d_inner_to_outer_undeform;
    dist_inner_to_tread_undef = data.d_inner_to_tread_undeform;
    curr_valid_mask = data.curr_valid_mask;
    prev_valid_mask = data.prev_valid_mask;
    contact_patch_mask = data.contact_patch_mask;
    
    mask_inner_def = inner_def(:,3) > 0.07;
    pcd_inner_def = pointCloud(inner_def); %pointCloud(inner_def(mask_inner_def,:));
    pcd_inner_undef = pointCloud(inner_undef);
    pcd_outer_def = pointCloud(outer_def(mask_inner_def,:));
    pcd_outer_undef = pointCloud(outer_undef);
    pcd_tread_def = pointCloud(tread_def);
    pcd_tread_undef = pointCloud(tread_undef);
    pcd_contact_patch = pointCloud(contact_patch);
    
    pcd_inner_def = rotate_cam_view(pcd_inner_def);
    pcd_inner_undef = rotate_cam_view(pcd_inner_undef);
    pcd_outer_def = rotate_cam_view(pcd_outer_def);
    pcd_outer_undef = rotate_cam_view(pcd_outer_undef);
    pcd_tread_def = rotate_cam_view(pcd_tread_def);
    pcd_tread_undef = rotate_cam_view(pcd_tread_undef);
    pcd_contact_patch = rotate_cam_view(pcd_contact_patch);
    
    input_data.dt = dt;
    input_data.time = time;
    input_data.pcd_inner_def = pcd_inner_def;
    input_data.pcd_inner_undef = pcd_inner_undef;
    input_data.pcd_outer_def = pcd_outer_def;
    input_data.pcd_outer_undef = pcd_outer_undef;
    input_data.pcd_tread_def = pcd_tread_def;
    input_data.pcd_tread_undef = pcd_tread_undef;
    input_data.pcd_contact_patch = pcd_contact_patch;
    input_data.dist_inner_def =  dist_inner_def; % dist_inner_def(mask_inner_def);
    input_data.dist_inner_to_outer_undef = dist_inner_to_outer_undef;
    input_data.dist_inner_to_tread_undef = dist_inner_to_tread_undef;
end 

function keyPressUpdate(fig, event)
    % Access stored data
    files = fig.UserData.files;
    idx = fig.UserData.currentIdx;
    hInner = fig.UserData.hInner;
    hOuter = fig.UserData.hOuter;
    ax = fig.UserData.ax;

    % Update index based on arrow keys
    if strcmp(event.Key, 'd')
        idx = min(idx + 1, length(files));
    elseif strcmp(event.Key, 'a')
        idx = max(idx - 1, 1);
    else
        return;
    end

    % Load new iteration
    data = load(fullfile(files(idx).folder, files(idx).name));
    inner = data.pcd_inner_deform;
    outer = data.pcd_outer_deform;

    % % Update scatter objects
    % hInner.XData = inner(:,1);
    % hInner.YData = inner(:,2);
    % hInner.ZData = inner(:,3);
    % 
    % hOuter.XData = outer(:,1);
    % hOuter.YData = outer(:,2);
    % hOuter.ZData = outer(:,3);

    % Update scatter objects
     % Parameters
    z_slice = 0.0;        % slice location
    tol = 0.001;          % thickness of slice (tolerance)
    
    % Mask points near that z
    mask_inner = abs(inner(:,1) - z_slice) < tol;
    mask_outer = abs(outer(:,1) - z_slice) < tol;

    
    hInner.XData = inner(mask_inner,2);
    hInner.YData = inner(mask_inner,3);

    hOuter.XData = outer(mask_outer,2);
    hOuter.YData = outer(mask_outer,3);

    % Update title
    title(ax, sprintf('Iteration %d', idx));

    % Save updated index
    fig.UserData.currentIdx = idx;
end

function track_rolling_points(data_iter, cmapName, chooseIdx, pc_model, files)
    row = 1;%480/2;
    column = 848/2;
    traj = zeros(length(data_iter), 3);
    deform = zeros(length(data_iter),1);
    idx = 848*row+column;
    for i = 1:length(data_iter)
        pc = data_iter{i}.pcd_inner_def;
           % pointCloud object
        pts = pc.Location;                    % Nx3 array
        traj(i,:) = pts(idx, :);  
        deform(i) = data_iter{i}.dist_inner_def(idx); % store trajectory
    end
    figure; grid on; axis equal;
    scatter3(traj(:,1), traj(:,2), traj(:,3), ...
        10, 'filled');
    hold on;
    scatter3(pts(:,1), pts(:,2), pts(:,3), ...
        1, 'k','filled');
    hold off;

    figure; grid on; axis equal;
    % scatter(traj(:,2),deform)
    scatter(traj(1:end-1,2), diff(traj(:,2)))
end

function create_3dscatter(inner, outer, currentIdx, files)
    % Create figure
    fig = figure('Name', 'Tyre Iterations', 'NumberTitle', 'off');
    ax = axes(fig);
    hold(ax, 'on');
    grid(ax, 'on'); axis(ax, 'equal');
    xlabel(ax, 'X'); ylabel(ax, 'Y'); zlabel(ax, 'Z');
    title(ax, sprintf('Iteration %d', currentIdx));
    
    % Create scatter objects once
    hInner = scatter3(ax, inner(:,1), inner(:,2), inner(:,3), 0.5);
    hOuter = scatter3(ax, outer(:,1), outer(:,2), outer(:,3), 0.5);
    
    % Store data and handles in figure's UserData for callback
    fig.UserData.files = files;
    fig.UserData.currentIdx = currentIdx;
    fig.UserData.hInner = hInner;
    fig.UserData.hOuter = hOuter;
    fig.UserData.ax = ax;
    
    % Key press callback
    fig.KeyPressFcn = @(src, event) keyPressUpdate(src, event);
end

function plot_longitudinal_circumferencial_angle_vs_deformation(data_iter, cmapName, chooseIdx, pc_model, files)
    figure;
    pcshow(data_iter{3}.pcd_outer_def,"BackgroundColor",[1,1,1])
    hold on;
    pcshow(data_iter{4}.pcd_outer_def,"BackgroundColor",[1,1,1])
    pcshow(pc_model,"BackgroundColor",[1,1,1])
    hold on;
    
    x_slices = [-0.10, -0.05, 0.00, 0.05, 0.10];   % choose your cross-section locations
    
    H = 0.4;    % height in Z direction
    W = 0.4;    % width in Y direction
    
    for i = 1:length(x_slices)
    
        x0 = x_slices(i);
    
        % Four corners of the plane
        p1 = [x0, -W,  H];
        p2 = [x0,  W,  H];
        p3 = [x0,  W, -H];
        p4 = [x0, -W, -H];
    
        % Draw the slicing plane
        patch('Vertices',[p1; p2; p3; p4], ...
              'Faces',[1 2 3 4], ...
              'FaceColor',[0 1 0.5], ...
              'FaceAlpha',0.25, ...
              'EdgeColor','k', ...
              'LineWidth',2);

        text(x0, -H-0.05, -H-0.05, sprintf('x = %.2f',x0), ...
          'FontSize',10, 'FontWeight','bold', 'Color','k');
    end
    
    hold off;
    grid on;
    axis equal;
    xlabel("x [m]")
    ylabel("y [m]")
    zlabel("z [m]")
    % title("Longitudinal Cross Sections of Inner Tyre Model with Outer Prediction ")
    figure; hold on; grid on;
    
    for i = 1:length(data_iter)
        pc = data_iter{i}.pcd_inner_def;
        S = data_iter{i}.dist_inner_def;
        pts = pc.Location;
        longitudinal_slice = 0.0;        % slice location
        tol = 0.001;          % thickness of slice (tolerance)
        
        % Mask points near that z
        mask_inner = abs(pts(:,1) - longitudinal_slice) < tol;
        % mask_outer = abs(outer(:,1) - z_slice) < tol;
    
        X = pts(mask_inner,1);
        Y = pts(mask_inner,2);
        Z = pts(mask_inner,3);
        
        theta = atan2(Z, Y);     % radians
        theta = unwrap(theta);   % avoid jumps at -pi/pi
        
        scatter(rad2deg(theta), -S(mask_inner)*1000, 3, 'filled');
    end
    ylim([-10,35])
    xlabel('\theta (deg)');
    ylabel('Deformation [mm]');
    title('Circumferential Angle vs Deformation');
    times_s = data_iter{1}.time(chooseIdx)/1000;
    legendStrings = arrayfun(@(t) sprintf('t = %.3f s', t), times_s, 'UniformOutput', false);
    legend(legendStrings)
    hold off;
end

function plot_radial_crossSection(data_iter, cmapName, chooseIdx, pc_model, files)
    figure;
    pcshow(data_iter{3}.pcd_outer_def,"BackgroundColor",[1,1,1])
    hold on;
    pcshow(data_iter{4}.pcd_outer_def,"BackgroundColor",[1,1,1])
    pcshow(pc_model,"BackgroundColor",[1,1,1])
    deg_arr = [-60,-65,-70,-75,-80,-85,-90,-95,-100,-105,-110,-115,-120];
    for i = 1:length(deg_arr)
        theta0 = deg2rad(90+deg_arr(i));
        n = [0, cos(theta0), -sin(theta0)];    
        v_radial = [0, -sin(theta0), -cos(theta0)];
        v_long   = [1, 0, 0];
        
        h = 0.42;  L = 0.2;
        center = [0 0 0];
        
        p1 = center +  L*v_long + h*v_radial;
        p2 = center -  L*v_long + h*v_radial;
        p3 = center -  L*v_long ;
        p4 = center +  L*v_long ;
        
        patch('Vertices',[p1; p2; p3; p4], ...
              'Faces',[1 2 3 4], ...
              'FaceColor',[0 1 0.5], ...
              'FaceAlpha',0.5, ...
              'EdgeColor','k', ...
              'LineWidth',1.5);
        text(-p1(1)-0.02,p1(2)+0.02,p1(3)+0.02, sprintf('\\theta = %.0f',rad2deg(theta0)-90), ...
          'FontSize',8, 'FontWeight','normal', 'Color','k','Rotation',rad2deg(theta0)-90);
        hold on;
    end
    hold off; grid on;
    xlabel("x [m]")
    ylabel("y [m]")
    zlabel("z [m]")
    theta0_deg = -60;
    tol_deg = 0.1;
    theta0 = deg2rad(theta0_deg);
    tol = deg2rad(tol_deg);
    figure;
    subplot(1,3,1)

    for i = 1:length(data_iter)
        pc = data_iter{i}.pcd_outer_def;
        S = data_iter{i}.dist_inner_def;
        % Extract point positions
        pts = pc.Location;
        Y = pts(:,2);
        Z = pts(:,3);
    
        % Compute circumferential angle
        theta = atan2(Z, Y);      % in radians
        % theta = unwrap(theta);    % avoid jump at +-pi
    
        % Mask near the target angle
        mask = abs(theta - theta0) < tol;
    
        % Output slice
        % pts_slice = pts(mask, :);
        scatter(pts(mask,1), sqrt(Y(mask).^2 + Z(mask).^2), 3, 'filled');
        hold on; grid on;
    end
    % ylim([-10,35])
    axis equal;
    xlabel('X [m]');
    ylabel('Radii [m]');
    title('Circumferential Angle vs Deformation');
    times_s = data_iter{1}.time(chooseIdx)/1000;
    legendStrings = arrayfun(@(t) sprintf('t = %.3f s', t), times_s, 'UniformOutput', false);
    legend(legendStrings)
    hold off;

    subplot(1,3,2)
    theta0_deg = -120;
    tol_deg = 0.1;
    theta0 = deg2rad(theta0_deg);
    tol = deg2rad(tol_deg);

    for i = 1:length(data_iter)
        pc = data_iter{i}.pcd_outer_def;
        S = data_iter{i}.dist_inner_def;
        % Extract point positions
        pts = pc.Location;
        Y = pts(:,2);
        Z = pts(:,3);
    
        % Compute circumferential angle
        theta = atan2(Z, Y);      % in radians
        % theta = unwrap(theta);    % avoid jump at +-pi
    
        % Mask near the target angle
        mask = abs(theta - theta0) < tol;
    
        % Output slice
        % pts_slice = pts(mask, :);
        scatter(pts(mask,1), sqrt(Y(mask).^2 + Z(mask).^2), 3, 'filled');
        hold on; grid on;
    end
    % ylim([-10,35])
    axis equal;
    xlabel('X [m]');
    ylabel('Radii [m]');
    title('Circumferential Angle vs Deformation');
    times_s = data_iter{1}.time(chooseIdx)/1000;
    legendStrings = arrayfun(@(t) sprintf('t = %.3f s', t), times_s, 'UniformOutput', false);
    legend(legendStrings)
    hold off;

    subplot(1,3,3)
    theta0_deg = -80;
    tol_deg = 0.1;
    theta0 = deg2rad(theta0_deg);
    tol = deg2rad(tol_deg);

    for i = 1:length(data_iter)
        pc = data_iter{i}.pcd_outer_def;
        S = data_iter{i}.dist_inner_def;
        % Extract point positions
        pts = pc.Location;
        Y = pts(:,2);
        Z = pts(:,3);
    
        % Compute circumferential angle
        theta = atan2(Z, Y);      % in radians
        % theta = unwrap(theta);    % avoid jump at +-pi
    
        % Mask near the target angle
        mask = abs(theta - theta0) < tol;
    
        % Output slice
        % pts_slice = pts(mask, :);
        % scatter(pts(mask,1), sqrt(Y(mask).^2 + Z(mask).^2), 3, 'filled');
        scatter(pts(mask,1), -S(mask), 3, 'filled');
        hold on; grid on;
    end
    % ylim([-10,35])
    axis equal;
    xlabel('X [m]');
    ylabel('Radii [m]');
    title('Circumferential Angle vs Deformation');
    times_s = data_iter{1}.time(chooseIdx)/1000;
    legendStrings = arrayfun(@(t) sprintf('t = %.3f s', t), times_s, 'UniformOutput', false);
    legend(legendStrings)
    hold off;
end

function plot_crossSection(inner, outer, currentIdx, files)
    % Parameters
    z_slice = 0.0;        % slice location
    tol = 0.001;          % thickness of slice (tolerance)
    
    % Mask points near that z
    mask_inner = abs(inner(:,1) - z_slice) < tol;
    mask_outer = abs(outer(:,1) - z_slice) < tol;

    % Extract slice points
    inner_slice = inner(mask_inner, :);
    outer_slice = outer(mask_outer, :);

    % Plot cross section (x vs y)
    fig = figure('Name', sprintf('Cross Section Iter %d', currentIdx));
    ax = axes(fig);
    hold(ax, 'on');
    grid(ax, 'on'); axis(ax, 'equal');
    xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]');
    title(ax, sprintf('Cross Section at z = %.3f (iteration %d)', z_slice, currentIdx));
    hInner = plot(inner_slice(:,2), inner_slice(:,3), 'b.', 'DisplayName', 'Inner Surface');
    hOuter = plot(outer_slice(:,2), outer_slice(:,3), 'r.', 'DisplayName', 'Outer Surface');
    legend show
    hold off

    % Store data and handles in figure's UserData for callback
    fig.UserData.files = files;
    fig.UserData.currentIdx = currentIdx;
    fig.UserData.hInner = hInner;
    fig.UserData.hOuter = hOuter;
    fig.UserData.ax = ax;
    fig.UserData.mask_inner = mask_inner;
    fig.UserData.mask_outer = mask_outer;
    
    % Key press callback
    fig.KeyPressFcn = @(src, event) keyPressUpdate(src, event);
end

function colorPointCloud(pc, S, cmapName)
    Smin = -0.055;
    Smax = 0;
    S_clipped = min(max(S, Smin), Smax);
    S_norm = (S_clipped - Smin) ./ (Smax - Smin);
    
    % cmap = feval(slanCM(cmapName), 256);
    cmap = slanCM(cmapName,256);
    idx = max(1, round(S_norm * 255));
    RGB_u8 = uint8(cmap(idx,:) * 255);
    
    pc.Color = RGB_u8;
end

function showPC_Colorbar_Hist(pc, S, cmapName, currentIdx, pc_model, files)
    Smin = -0.035;  
    Smax = 0.01;

    cmap = slanCM(cmapName,256);

    Sclip = min(max(S, Smin), Smax);
    S_norm = (Sclip - Smin) / (Smax - Smin);
    idx = min(max(round(S_norm*255)+1,1),256);
    col = cmap(idx,:);

    fig = figure('Position',[100 100 900 600]);
    
    ax1 = subplot(1,2,1);

    pts = pc.Location;
    hScatter = scatter3(ax1, pts(:,1), pts(:,2), pts(:,3), ...
        10, col, 'filled');

    hold(ax1,'on');

    % % plot model cloud
    % ptsM = pc_model.Location;
    % scatter3(ax1, ptsM(:,1), ptsM(:,2), ptsM(:,3), ...
    %     5, [0 0 0], 'filled');

    axis(ax1,'equal')
    xlabel(ax1,'X'), ylabel(ax1,'Y'), zlabel(ax1,'Z')
    title(ax1,'3D Point Cloud')
    colormap(ax1, cmap)
    clim(ax1,[Smin Smax])
    cb = colorbar(ax1,'Location','eastoutside');
    % cb.Label.String = 'Signed deformation';
    cb.FontSize = 11;
    cb.RulerLocation = 'left';

    hold(ax1,'off')

    ax3 = subplot(1,2,2);

    nbins = 1000;
    [counts, edges] = histcounts(S, nbins);
    binCenters = (edges(1:end-1) + edges(2:end))/2;

    S_norm_bin = (binCenters - Smin) / (Smax - Smin);
    idx_bins = min(max(round(S_norm_bin*255)+1,1),256);

    patchHandles = gobjects(nbins,1);

    hold(ax3,'on')
    for i = 1:nbins
        x = [0 counts(i) counts(i) 0];
        y = [edges(i) edges(i) edges(i+1) edges(i+1)];
        patchHandles(i) = patch(ax3, x, y, cmap(idx_bins(i),:), ...
            'EdgeColor','none');
    end
    hold(ax3,'off')

    xlabel(ax3,'Count')
    ylim(ax3,[Smin Smax])
    title(ax3,'Signed Deformation','FontSize',11)
    ax3.YTickLabel = [];
    ax1.Position = [0.05 0.1 0.6 0.8];
    ax3.Position = [0.724 0.1 0.1 0.8];

    data.files = files;
    data.currentIdx = currentIdx;

    data.Smin = Smin;
    data.Smax = Smax;
    data.cmap = cmap;
    data.nbins = nbins;

    data.hScatter = hScatter;
    data.ax1 = ax1;
    data.ax3 = ax3;
    data.patchHandles = patchHandles;

    fig.UserData = data;

    fig.WindowKeyPressFcn = @(src,event) keyPressUpdate_3(src,event);

end


function keyPressUpdate_3(fig, event)

    data = fig.UserData;

    % step index
    if strcmp(event.Key,'m')
        data.currentIdx = min(data.currentIdx + 1, numel(data.files));
    elseif strcmp(event.Key,'n')
        data.currentIdx = max(data.currentIdx - 1, 1);
    else
        return
    end

    input = load_data(data.currentIdx,data.files);   % <-- user function
    pts = input.pcd_inner_def.Location;
    S   = input.dist_inner_def;

    Sclip = min(max(S, data.Smin), data.Smax);
    S_norm = (Sclip - data.Smin) / (data.Smax - data.Smin);
    idx = min(max(round(S_norm*255)+1,1),256);
    col = data.cmap(idx,:);

    set(data.hScatter, ...
        'XData', pts(:,1), ...
        'YData', pts(:,2), ...
        'ZData', pts(:,3), ...
        'CData', col);

    % update title
    title(data.ax1, sprintf('3D Point Cloud – Iter %d', data.currentIdx));

    [counts, edges] = histcounts(S, data.nbins);
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    S_norm_bin = (binCenters - data.Smin) / (data.Smax - data.Smin);
    idx_bins = min(max(round(S_norm_bin*255)+1,1),256);

    for i = 1:data.nbins
        x = [0 counts(i) counts(i) 0];
        y = [edges(i) edges(i) edges(i+1) edges(i+1)];

        h = data.patchHandles(i);
        h.XData = x;
        h.YData = y;
        h.FaceColor = data.cmap(idx_bins(i),:);
    end

    % save updated data
    fig.UserData = data;
end


function pcd_r = rotate_cam_view(pcd)
   T = zeros(4,4);
    T(4,4) = 1;  % homogeneous
    
    % --- Define rotation angle ---
    alpha = 19.2 * pi/180;  % convert degrees to radians
    
    % --- Define translation vector ---
    x_translate = 0;
    y_translate = 0.2162 + 0.07254;
    z_translate = 0.0393;
    xyz_translate = [x_translate, y_translate, z_translate];
    
    % --- Define rotation matrix about X-axis ---
    Rx = [1, 0, 0; 
          0, cos(alpha), -sin(alpha); 
          0, sin(alpha),  cos(alpha)];
    
    theta = 204*(pi/180);
    Rx_centering = [1, 0, 0;
                    0, cos(theta), -sin(theta);
                    0, sin(theta),  cos(theta)];
    
    % --- Display translation and rotation ---
    % disp('Translation:'); disp(xyz_translate);
    % disp('Rotation matrix:'); disp(Rx);
    
    % --- Assuming you have pointCloud objects ---
    % tracked_t2cam = pointCloud(t2cam_points);  % Nx3
    % model_pcd = pointCloud(model_points);
    
    % --- Apply translation ---
    
    pt = pcd.Location + [0, 0.07254, 0.0393]; 
    
    % --- Apply rotation about a specific center (0, 0.2162, 0) ---
    % center = [0, 0.07254, 0.0393];
    % % Move points so center is origin
    % pt = pt - center; 
    % Apply rotation
    pt = pt * Rx'; 
    % Move points back
    pt = pt + [0,0.2162,0]; 
    % Update point cloud
    pt = pt * Rx_centering';
    pcd_r = pointCloud(pt);
end

function model_r = rectify_model(ptCloud)
    scale = 0.03787;  % replace with your scale

    % --- Scale points about the origin ---
    pts = ptCloud.Location;       % Nx3
    pts = pts * scale;            % scale manually
    
    % --- Compute centroid and translate points to origin ---
    centroid = [0.42841208, -1.6929364, 3.6547658] * scale;
    pts = pts - centroid;
    
    % --- First rotation ---
    R = [-0.03135062 , -0.03143202 , -0.9990141;
          0.35852575 ,  0.9326368  , -0.04059484;
          0.93299323 , -0.35944495 , -0.01796954];  % Transpose done here
    pts = (R * pts')';  % rotate around origin
    
    % --- Second translation ---
    translate_vec = [0, 0.0948+0.0355+0.0139, 0.0944-0.0216+0.0109] * scale;
    pts = pts - translate_vec;
    
    % --- Second rotation ---
    R2 = [-1, 0, 0;
           0, 1, 0;
           0, 0, -1];
    pts = (R2 * pts')';
    
    % --- Update point cloud ---
    model_r = pointCloud(pts);
    model_r.Color = ptCloud.Color;
    
end

function plot_radial_displacement(model_pcd)
    % Accept either Nx3 matrix or pointCloud object
    if isa(model_pcd, 'pointCloud')
        pts = model_pcd.Location;
    else
        pts = model_pcd;
    end
    
    % Ensure Nx3
    if size(pts,2) ~= 3
        error('Input must be Nx3 or a pointCloud');
    end

    % Extract coordinates
    x = pts(:,1);
    y = pts(:,2);   % tyre width direction
    z = pts(:,3);

    % Compute radial distance in XY plane
    radial_dist = sqrt(y.^2 + z.^2);

    % Compute angular position θ
    theta = atan2(y, z);           % radians
    theta_deg = rad2deg(theta);    % convert to degrees

    % Scatter plot: X = θ, Y = tyre width, color = radial displacement
    figure;
    scatter(theta_deg, x, 8, radial_dist, 'filled');
    xlabel('Angle θ (degrees)');
    ylabel('Tyre width (Y coordinate)');
    title('Radial Displacement Map (Color = Radius)');
    colormap jet;
    colorbar;
    grid on;
    % clim([0.39 0.4]);
    clim([0.345 0.36]);

    % Optional: wrap angle nicely
    xlim([-180 180]);
end