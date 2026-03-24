% clear all

% Folder with saved iterations
folder = 'D:\stored_arrays\Steering_test_9';  %'saved_arrays';
% folder = 'D:\stored_arrays\Rolling_back_test';
files = dir(fullfile(folder, 'iteration_*.mat'));  % or .npz converted to .mat
nFiles = length(files);
%%
% Start iteration
% currentIdx = 1;
% inner_model_pcd = pcread('full_outer_inner_smoothed_part_only.ply');
inner_model_pcd = pcread('full_outer_inner_part_only.ply');
inner_model_pcd = rectify_model(inner_model_pcd);
outer_model_pcd = pcread('full_outer_outer_part_only.ply');
outer_model_pcd = rectify_model(outer_model_pcd);
%%

chooseIdx = %70:5:295; %146:157; %70:5:295;%1:2:600;%140:195; %70:120; %118:140;%170:1:210; %175:1:201; %[80,90,100,110,120,130,140,150]; %110:1:206; %[70,98]; %70:5:295; %[1,50:210]; %[1,75:92];%[155:210]; %[146:1:146+1*13]; %110:1:206; %[210];%1:560; %[1,160,165,170,180,190,210]; [380];
data = load_data(chooseIdx,files); 
%%
figure(1)
plot(data{1}.time(2:560-28)/1000,vel_est_test_7(29:end),'DisplayName','Load 4000N');
hold on;
plot(data{1}.time(2:560-23)/1000,vel_est_test_8(24:end),'DisplayName','Load 5000N');
plot(data{1}.time(2:560)/1000,vel_est_test_9(1:end),'DisplayName','Load 6000N');
plot(data{1}.time(2:560-91)/1000,vel_est_test_21(92:end),'DisplayName','Load 7000N');
hold off;
xlabel("Time (s)")
ylabel("Velocity (m/s)")
grid on
legend()
xlim([0,30])
title("Max Tyre Contact Velocity Response to STTR Plate Velocity at 16.67mm/s")

%%

% plot_one_data(data)
% vel_est_test_8 = plot_vel(data);
% plot_pre_data(data)
% track_rolling_points(data);
% plot_hist_time(data)
t7  = [6400];
t8 = [7030];
t9 = [7790];
t21 = [8360];

% plot_max_def_time(data)
% plot_time_series_def(data);
overlay_contact_patch(data);
overlay_contact_patch_t(data);
% plot_cross_sections(data);
% plot_cross_sections_s(data,data_s);
% track_cross_sections(data);
% plot_contourf(data);
% vis_vel_field(data);

% vid_path = 'C:\Users\amalp\Desktop\MSS732\projected\DATA\test_9_sine_0_2Hz_3_cycles_80000_right_tyre_d3_c18_2bar\GOPR1038.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\test_9_sine_0_2Hz_3_cycles_80000_right_tyre_d3_c18_2bar\GOPR1038.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\test_4_sine_0_2Hz_3_cycles_40000_right_tyre_d3_c18_2bar\GOPR1033.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\test_24_(22_dspace)_square_tubing_backward_roll_release_d3_c18_2bar\GOPR1054.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\rolling_test_right_tyre_full_rotation_backwards_d3_c18_2bar_test2\IMG_3850.MOV';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_21_1000mm_min_700kg_10sec_d3_c18_2bar\GOPR1267.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_9_1000mm_min_600kg_10sec_d3_c18_2bar\GOPR1252.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_7_1000mm_min_400kg_10sec_d3_c18_2bar\GOPR1249.MP4';
%'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_8_1000mm_min_500kg_10sec_d3_c18_2bar\GOPR1250.MP4';

% 'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_13_cleat_1500mm_min_400kg_10sec_d3_c18_2bar\GOPR1258.MP4';
% 'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_14_cleat_1500mm_min_500kg_10sec_d3_c18_2bar\GOPR1259.MP4';
% 'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_15_cleat_1500mm_min_600kg_10sec_d3_c18_2bar\GOPR1260.MP4';

% out_folder = 'C:\Users\amalp\Desktop\MSS732\projected\istvs_images\MIR_figs\Steering_11'; %'C:\Users\amalp\Desktop\MSS732\projected\DATA\STTR_test_8_1000mm_min_500kg_10sec_d3_c18_2bar';
% extract_frames_from_mp4(vid_path,chooseIdx,out_folder)

% function plot_max_def_time(data)
% N = length(data)
% max_def_t = zeros(N,1);
%     for i = 1:N
%         max_def_t(i,1) = min(data{i}.dist_inner_def(find(data{i}.contact_patch_mask)));
%     end
%     plot(max_def_t)
% end

function plot_max_def_time(data)

N = length(data);
max_def_t = zeros(N,1);

for i = 1:N
    mask = data{i}.pcd_inner_def.Location(:,3) < -0.32 & data{i}.pcd_inner_def.Location(:,3) > -0.36;
    vals = data{i}.dist_inner_def(mask);
    
    % Use 5th percentile instead of minimum
    max_def_t(i) = min(vals );
    % max_def_t(i) = prctile(vals,1);  
end

plot(max_def_t)
xlabel('Frame')
ylabel('Max Deformation (mm)')
title('Robust Maximum Deformation vs Time')

end

function vis_vel_field(data)
    N = length(data);
    disp_fields_unmasked = cell(N-1,1);
    disp_fields = cell(N-1,1);
    vel_est = zeros(N-1,1);

    % Extract time vector (same for all entries)
    dt = data{1}.dt/1000;          % dt is length N-1
    time_vec = data{1}.time/1000;  % N×1 time vector

    % traj = zeros(N-1, 3);
    % u = zeros(N-1, 3);
    % d_traj = zeros(N-1, 3);
    % v_traj = zeros(N-1, 3);
    % deform_x = zeros(N-1,1);
    % deform_y = zeros(N-1,1);
    % deform_z = zeros(N-1,1);
    % undef = zeros(N-1, 3);
    % def = zeros(N-1, 3);

    for i = 2:N
        pts = data{i}.pcd_inner_def.Location;
        pts_curr = data{i}.pcd_inner_def.Location; %(data{i}.curr_valid_mask & data{i}.prev_valid_mask,:); %& data{i}.contact_patch_mask
        pts_prev = data{i-1}.pcd_inner_def.Location; %(data{i}.curr_valid_mask & data{i}.prev_valid_mask,:); %& data{i}.contact_patch_mask
        % u(i-1,:) = data{i-1}.dist_3d;
        u = pts_curr - pts_prev;
        % undef(i-1,:) = data{i-1}.pcd_inner_undef.Location(idx,:);
        % def(i-1,:) = data{i-1}.pcd_inner_def.Location(idx,:);
        % traj(i-1,:) = pts_prev(idx,:);
        % disp_fields{i-1} = pts_curr - pts_prev;                   % Nx3 array
        % d_traj(i-1,:) = disp_fields{i-1}(idx,:);  
        % v_traj(i-1,:) = disp_fields{i-1}(idx,:)./(time_vec(i-1));
        mask = pts(:,3) < -0.3;
        figure(1); clf
        
        
        % --- SUBPLOT 1: Ux ---
        subplot(1,3,1)
        title("U_x")
        scatter3(pts(mask,1), pts(mask,2), pts(mask,3), 3, u(mask,2), 'filled');
        xlabel("Y [m]"); ylabel("X [m]"); zlabel("Z [m]");
        grid on; view(0,90); daspect([1 1 1]);
        clim([-0.005, 0.005]);
        
        % --- SUBPLOT 2: Uy ---
        subplot(1,3,2)
        title("U_y")
        scatter3(pts(mask,1), pts(mask,2), pts(mask,3), 3, u(mask,1), 'filled');
        xlabel("Y [m]"); ylabel("X [m]"); zlabel("Z [m]");
        grid on; view(0,90); daspect([1 1 1]);
        clim([-0.005, 0.005]);
        
        % --- SUBPLOT 3: Uz ---
        subplot(1,3,3)
        title("U_z")
        scatter3(pts(mask,1), pts(mask,2), pts(mask,3), 3, u(mask,3), 'filled');
        xlabel("Y [m]"); ylabel("X [m]"); zlabel("Z [m]");
        grid on; view(0,90); daspect([1 1 1]);
        clim([-0.005, 0.005]);
        
        % --- SHARED COLORBAR ---
        colormap(parula);
        
        % create a tiny, invisible axes ONLY for the colorbar
        axR = axes('Position',[0 0 1 1],'Visible','off');
        cR = colorbar(axR, 'southoutside');
        clim([-5, 5]);
        % cR.Ticks = linspace(-0.005, 0.005, 5);   % for example 5 ticks
        % cR.TickLabels = arrayfun(@(x) sprintf('%.3f', x), cR.Ticks, 'UniformOutput', false);
        % adjust colorbar position (x, y, width, height)
        cR.Position = [0.35 0.07 0.30 0.03];
        
        % link colorbar to current figure colormap
        set(get(cR,'Label'),'String','Displacement [mm]');
        sgtitle("Displacement Tracking [U_x,U_y,U_z]")
    end
  
    % figure; 
    % scatter3(traj(:,1), traj(:,2), traj(:,3), ...
    %     10, 'filled');
    % grid on; axis equal;
    % hold on;
    % % scatter3(d_traj(:,1), d_traj(:,2), d_traj(:,3),...
    % %     1, 'k','filled');
    % scatter3(pts_prev(:,1), pts_prev(:,2), pts_prev(:,3), ...
    %     1, 'k','filled');
    % hold off;
    % xlabel("X [m]")
    % ylabel("Y [m]")
    % zlabel("Z [m]")
    % 
    % figure; 
    % disp(size(d_traj))
    % disp(size(time_vec(110:210)))
    % % scatter(traj(:,2),deform)
    % % plot(time_vec(110:205),v_traj(:,1))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:205),v_traj(:,2))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:205),v_traj(:,3))
    % % plot(time_vec(110:204),strain(:,1))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:204),strain(:,2))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:204),strain(:,3))
    % % plot(time_vec(110:205),u(:,1))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:205),u(:,2))
    % % figure; grid on; axis equal;
    % % plot(time_vec(110:205),u(:,3))
    % plot(time_vec(115:205),strain_theta(6:end))
    % xlabel("Time [s]")
    % ylabel("Strain_{\theta}")
    % title("Longitudinal Strain of Tracked Point over time")
    % grid on;
    % figure; 
    % plot(s_cum(5:end),strain_theta(7:end))
    % % xlabel("Time [s]")
    % % ylabel("Strain_{\theta}")
    % % title("Longitudinal Strain of Tracked Point over time")
    % grid on;
    % figure; 
    % plot(rad2deg(theta_undef(5:end)),movmean(strain_theta(5:end),10))
    % xlabel("Degrees \degrees")
    % ylabel("Strain_{\theta}")
    % title("Longitudinal Strain of Tracked Point over Arc")
    % grid on;
    % figure; 
    % plot(time_vec(110:205),u_theta)
    % xlabel("Time [s]")
    % ylabel("Deformation_{\theta}")
    % title("Longitudinal Deformation of Tracked Point over time")
    % grid on;
end


function track_cross_sections(data)
    N = length(data);
    disp_fields_unmasked = cell(N-1,1);
    disp_fields = cell(N-1,1);
    vel_est = zeros(N-1,1);

    % Extract time vector (same for all entries)
    dt = data{1}.dt/1000;          % dt is length N-1
    time_vec = data{1}.time/1000;  % N×1 time vector

    % row = 0:479;
    % column = 424;
    % idx = 848*(row)+column;
    % undef = zeros(N-1, 3);
    % def = zeros(N-1, 3);

    % rows   = 1:479;
    % column = 848/2;
    % idx = row + (column - 1) * 480;
    H=480;
    W=848;
    colors = parula(N);
    idx_numpy = reshape(1:H*W, H, W);  % MATLAB column-major
    idx_numpy = idx_numpy';            % transpose → matches NumPy row-major
    idx_numpy = idx_numpy(:);          % flatten
    
    column = 100:10:700;      % center column (example)
    row = (1:H).';     % all rows as column vector

    idx = (column-1)*H + row;         % correct MATLAB indexing
    
    figure; hold on
    for i = 1:N
        col = colors(i,:);
    
        pts = data{i}.pcd_inner_def.Location;
        ptso = data{i}.pcd_outer_def.Location;
        % FIX ORDERING
        pts_i = zeros(size(pts));
        pts_i(idx_numpy, :) = pts;
        pts_o = zeros(size(ptso));
        pts_o(idx_numpy, :) = ptso;
        mask_i = pts_i(idx,3) < -0.3;
        mask_o = pts_o(idx,3) < -0.3;
        scatter3(pts_i(idx(mask_i),1), pts_i(idx(mask_i),2), pts_i(idx(mask_i),3), ...
                 1, col, "filled");
        scatter3(pts_o(idx(mask_o),1), pts_o(idx(mask_o),2), pts_o(idx(mask_o),3), ...
                 1, col, "filled");


        % scatter3(pts_o(idx,1), pts_o(idx,2), pts_o(idx,3), 1, col, "filled");
        % scatter3(pts([1, 480, 480*848-480+1, 480*848], 1),pts([1, 480, 480*848-480+1, 480*848], 2),pts([1, 480, 480*848-480+1, 480*848], 3),19)
        % scatter3()
    end
    % scatter3(pts(:,1),pts(:,2),pts(:,3),0.1)
    % daspect([1 1 1]);
    colormap(parula(N))
    colorbar
    grid on

end

function plot_contourf(data)
    % Example data
    inner_pts = data{1}.pcd_inner_def.Location;   
    tread_pts = data{1}.pcd_tread_def.Location; %data{1}.pcd_tread_def.Location; % Nx3
    
    deformation = data{1}.dist_inner_def;  % Nx1
    
    % Define bins (edges) and colormap
    numBins = 10; 
    edges = linspace(-0.035, 0.0, numBins+1);  % 10 bins
    cmap = slanCM('plasma', numBins-3);          % exactly numBins colors!
    
    
    % Assign each point to a bin
    binIdx = discretize(deformation, edges); 
    binIdx(isnan(binIdx) & deformation < edges(1)) = 1;
    binIdx(isnan(binIdx) & deformation > edges(end)) = numBins;
    
    mask = inner_pts(:,3) < -0.3;
    maskt = tread_pts(:,3) < -0.3;

    figure;
    
    % ---------------- Inner Point Cloud ----------------
    subplot(1,2,1)
    scatter3(inner_pts(mask,1), inner_pts(mask,2), inner_pts(mask,3), 5, deformation(mask), 'filled');
    axis equal; grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Inner Point Cloud');
    view(0, 90)
    
    colormap(cmap);
    clim([-0.035 0.0])
    % caxis([1 numBins]);             % important: match bin indices
    c1 = colorbar;
    % c1.Ticks = 10;           % tick per bin
    % c1.TickLabels = round(edges(1:end),3); % show left edge of each bin
    c1.Label.String = 'Deformation [m]';
    
    % ---------------- Tread Point Cloud ----------------
    subplot(1,2,2)
    scatter3(tread_pts(maskt,1), tread_pts(maskt,2), tread_pts(maskt,3), 5, deformation(maskt), 'filled');
    axis equal; grid on;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Tread Point Cloud');
    view(0, 90)
    
    colormap(cmap);
    clim([-0.035 0.0])
    c2 = colorbar;
    % c2.Ticks = 1:numBins+1;
    % c2.TickLabels = round(edges(1:end),3);
    c2.Label.String = 'Deformation [m]';
    sgtitle('Static Steering angle 11\circ test: Contact Patch Prediction')
    % sgtitle(sprintf('Contact Patch Prediction from t = %.1fs',data{1}.time(250)/1000))
end

function plot_cross_sections_s(data,data_s)

    theta0_deg = [-80,-90,-100];
    long_x = [-0.05,0,0.05];
    tol_deg = 0.2;
    tol = deg2rad(tol_deg);

    figure;

    % build alpha values (earlier timesteps more transparent)
    N = length(data);
    alphas = linspace(0.15, 1, N);    % adjustable
    % blueC = [0 0.4470 0.7410];
    % redC = [0.8500 0.3250 0.0980];

    cmapB = winter(N);
    cmapR = autumn(N);

    for i = [N]
        alpha_t = alphas(i);

        inner_def_pts   = data{i}.pcd_inner_def.Location;
        inner_undef_pts = data{i}.pcd_inner_undef.Location;
        outer_def_pts   = data{i}.pcd_outer_def.Location;
        outer_undef_pts = data{i}.pcd_outer_undef.Location;

        theta_inner_def   = atan2(inner_def_pts(:,3),inner_def_pts(:,2));
        theta_inner_undef = atan2(inner_undef_pts(:,3),inner_undef_pts(:,2));
        theta_outer_def   = atan2(outer_def_pts(:,3),outer_def_pts(:,2));
        theta_outer_undef = atan2(outer_undef_pts(:,3),outer_undef_pts(:,2));

        inner_def_pts_s   = data_s{i}.pcd_inner_def.Location;
        inner_undef_pts_s = data_s{i}.pcd_inner_undef.Location;
        outer_def_pts_s   = data_s{i}.pcd_outer_def.Location;
        outer_undef_pts_s = data_s{i}.pcd_outer_undef.Location;

        theta_inner_def_s   = atan2(inner_def_pts_s(:,3),inner_def_pts_s(:,2));
        theta_inner_undef_s = atan2(inner_undef_pts_s(:,3),inner_undef_pts_s(:,2));
        theta_outer_def_s   = atan2(outer_def_pts_s(:,3),outer_def_pts_s(:,2));
        theta_outer_undef_s = atan2(outer_undef_pts_s(:,3),outer_undef_pts_s(:,2));

        blueC = cmapB(i,:);
        redC = cmapR(i,:);

        % -------------------------
        %  RADIAL CROSS SECTIONS
        % -------------------------
        for j = 1:length(theta0_deg)
            theta0 = deg2rad(theta0_deg(j));

            mask_inner_def   = abs(theta_inner_def   - theta0) < tol;
            mask_inner_undef = abs(theta_inner_undef - theta0) < tol;
            mask_outer_def   = abs(theta_outer_def   - theta0) < tol;
            mask_outer_undef = abs(theta_outer_undef - theta0) < tol;

            mask_inner_def_s   = abs(theta_inner_def_s   - theta0) < tol;
            mask_inner_undef_s = abs(theta_inner_undef_s - theta0) < tol;
            mask_outer_def_s   = abs(theta_outer_def_s   - theta0) < tol;
            mask_outer_undef_s = abs(theta_outer_undef_s - theta0) < tol;

            subplot(3,2,2*j)
            hold on

            h1 = scatter(inner_def_pts(mask_inner_def,1), ...
                    -sqrt(inner_def_pts(mask_inner_def,2).^2 + inner_def_pts(mask_inner_def,3).^2), ...
                    5,[0,1,0], 'filled');

            h2 = scatter(inner_def_pts_s(mask_inner_def_s,1), ...
                    -sqrt(inner_def_pts_s(mask_inner_def_s,2).^2 + inner_def_pts_s(mask_inner_def_s,3).^2), ...
                    5,[0,0,1], 'filled');

            h3 = scatter(inner_undef_pts(mask_inner_undef,1), ...
                    -sqrt(inner_undef_pts(mask_inner_undef,2).^2 + inner_undef_pts(mask_inner_undef,3).^2), ...
                    5,[0,0,0], 'filled');


            % scatter(inner_undef_pts(mask_inner_undef,1), ...
            %         -sqrt(inner_undef_pts(mask_inner_undef,2).^2 + inner_undef_pts(mask_inner_undef,3).^2), ...
            %         3, 'filled', ...
            %         'MarkerFaceAlpha', alpha_t);

            scatter(outer_def_pts(mask_outer_def,1), ...
                    -sqrt(outer_def_pts(mask_outer_def,2).^2 + outer_def_pts(mask_outer_def,3).^2), ...
                    5, [0,1,0],'filled');

            scatter(outer_def_pts_s(mask_outer_def_s,1), ...
                    -sqrt(outer_def_pts_s(mask_outer_def_s,2).^2 + outer_def_pts_s(mask_outer_def_s,3).^2), ...
                    5, [0,0,1],'filled');

            scatter(outer_undef_pts(mask_outer_undef,1), ...
                    -sqrt(outer_undef_pts(mask_outer_undef,2).^2 + outer_undef_pts(mask_outer_undef,3).^2), ...
                    5, [0,0,0],'filled');

            % scatter(outer_undef_pts(mask_outer_undef,1), ...
            %         -sqrt(outer_undef_pts(mask_outer_undef,2).^2 + outer_undef_pts(mask_outer_undef,3).^2), ...
            %         3, 'filled', ...
            %         'MarkerFaceAlpha', alpha_t);

            ylim([-0.4,-0.28])
            xlim([-0.15,0.15])
            daspect([1 1 1]);
            grid on
            xlabel('Y [m]');
            ylabel('Radii [m]');
            title(sprintf('Radial Cross Section at \\theta = %.0f°', theta0_deg(j)));
        end

        % -------------------------
        %  LONGITUDINAL CROSS SECTIONS
        % -------------------------
        for k = 1:length(long_x)
            x_slice = long_x(k);

            mask_inner_def   = abs(inner_def_pts(:,1) - x_slice) < tol & inner_def_pts(:,3) < -0.3;
            mask_inner_undef = abs(inner_undef_pts(:,1) - x_slice) < tol & inner_undef_pts(:,3) < -0.3;
            mask_outer_def   = abs(outer_def_pts(:,1) - x_slice) < tol & outer_def_pts(:,3) < -0.3;
            mask_outer_undef = abs(outer_undef_pts(:,1) - x_slice) < tol & outer_undef_pts(:,3) < -0.3;

            mask_inner_def_s   = abs(inner_def_pts_s(:,1) - x_slice) < tol & inner_def_pts_s(:,3) < -0.3;
            mask_inner_undef_s = abs(inner_undef_pts_s(:,1) - x_slice) < tol & inner_undef_pts_s(:,3) < -0.3;
            mask_outer_def_s   = abs(outer_def_pts_s(:,1) - x_slice) < tol & outer_def_pts_s(:,3) < -0.3;
            mask_outer_undef_s = abs(outer_undef_pts_s(:,1) - x_slice) < tol & outer_undef_pts_s(:,3) < -0.3;


            subplot(3,2,2*k-1)
            hold on

            scatter(inner_def_pts(mask_inner_def,2), inner_def_pts(mask_inner_def,3), ...
                    5, [0,1,0],'filled');

            scatter(inner_def_pts_s(mask_inner_def_s,2), inner_def_pts_s(mask_inner_def_s,3), ...
                    5, [0,0,1],'filled');

            scatter(inner_undef_pts(mask_inner_undef,2), inner_undef_pts(mask_inner_undef,3), ...
                    5, [0,0,1],'filled');
            
            % scatter(inner_undef_pts(mask_inner_undef,2), inner_undef_pts(mask_inner_undef,3), ...
            %         3, 'filled', 'MarkerFaceAlpha', alpha_t);

            scatter(outer_def_pts(mask_outer_def,2), outer_def_pts(mask_outer_def,3), ...
                    5,[0,1,0],'filled');

            scatter(outer_def_pts_s(mask_outer_def_s,2), outer_def_pts_s(mask_outer_def_s,3), ...
                    5, [0,0,1],'filled');

            scatter(outer_undef_pts(mask_outer_undef,2), outer_undef_pts(mask_outer_undef,3), ...
                    5, [0,0,0],'filled');

            % scatter(outer_undef_pts(mask_outer_undef,2), outer_undef_pts(mask_outer_undef,3), ...
            %         3, 'filled', 'MarkerFaceAlpha', alpha_t);

            xlim([-0.22,0.22])
            daspect([1 1 1]);
            grid on
            xlabel('X [m]');
            ylabel('Y [m]');
            title(sprintf('Circumferential Cross Section at x = %.2f m', x_slice));
        end
    end
    sgtitle('Cross Sections: STTR Flat plate vs. Cleat at 6000N');
    axR = axes('Position',[0 0 1 1],'Visible','off'); % invisible axis
    legend([h1 h2 h3], {'Flat Plate','Square Cleat', 'Undeformed'});
    % colormap(axR, cmapR);
    % 
    % cR = colorbar(axR, 'eastoutside');
    % cR.Position = [0.9 0.1 0.015 0.8];  % below figure
    % % cR.Label.String = 'Red metric';
    % % cR.AxisLocation = 'out';
    % cR.TickLabels = [];
    % 
    % % ---- Blue colormap colorbar ----
    % axB = axes('Position',[0 0 1 1],'Visible','off'); % invisible axis
    % colormap(axB, cmapB);
    % 
    % cB = colorbar(axB, 'eastoutside');
    % cB.Position = [0.92 0.1 0.015 0.8]; % placed below the first one
    % cB.Label.String = 'Deformation over time [s]';
    % cB.AxisLocation = 'out';
    % totalTime = length(data) * 0.067;    % total time in seconds
    % clim(axB, [0 totalTime]);            % color axis in seconds
    % 
    % % Choose tick positions along the colorbar in seconds
    % numTicks = min(length(data), 6);     % max 6 ticks to avoid crowding
    % tickPositions = linspace(0, totalTime, numTicks);
    % 
    % cB.Ticks = tickPositions;            % positions along CLim
    % cB.TickLabels = round(tickPositions, 2);
    
    % Set consistent limits
    % clim(axR, [1 length(data)]);
    % clim(axB, [0 N*0.067]);
    % Legend only once
    % lgd = legend({'Inner Def','Inner Undef','Outer Def','Outer Undef'});
    % lgd.Orientation = 'horizontal';
end


function plot_cross_sections(data)

    theta0_deg = [-80,-90,-100];
    long_x = [-0.05,0,0.05];
    tol_deg = 0.1;
    tol = deg2rad(tol_deg);

    figure;

    % build alpha values (earlier timesteps more transparent)
    N = length(data);
    alphas = linspace(0.15, 1, N);    % adjustable
    % blueC = [0 0.4470 0.7410];
    % redC = [0.8500 0.3250 0.0980];

    cmapB = winter(N);
    cmapR = autumn(N);

    for i = 1:N
        alpha_t = alphas(i);

        inner_def_pts   = data{i}.pcd_inner_def.Location;
        inner_undef_pts = data{i}.pcd_inner_undef.Location;
        outer_def_pts   = data{i}.pcd_outer_def.Location;
        outer_undef_pts = data{i}.pcd_outer_undef.Location;

        theta_inner_def   = atan2(inner_def_pts(:,3),inner_def_pts(:,2));
        theta_inner_undef = atan2(inner_undef_pts(:,3),inner_undef_pts(:,2));
        theta_outer_def   = atan2(outer_def_pts(:,3),outer_def_pts(:,2));
        theta_outer_undef = atan2(outer_undef_pts(:,3),outer_undef_pts(:,2));

        blueC = cmapB(i,:);
        redC = cmapR(i,:);

        % -------------------------
        %  RADIAL CROSS SECTIONS
        % -------------------------
        for j = 1:length(theta0_deg)
            theta0 = deg2rad(theta0_deg(j));

            mask_inner_def   = abs(theta_inner_def   - theta0) < tol;
            mask_inner_undef = abs(theta_inner_undef - theta0) < tol;
            mask_outer_def   = abs(theta_outer_def   - theta0) < tol;
            mask_outer_undef = abs(theta_outer_undef - theta0) < tol;

            subplot(3,2,2*j)
            hold on
            

            scatter(inner_def_pts(mask_inner_def,1), ...
                    -sqrt(inner_def_pts(mask_inner_def,2).^2 + inner_def_pts(mask_inner_def,3).^2), ...
                    2,redC, 'filled', ...
                    'MarkerFaceAlpha', alpha_t);

            % scatter(inner_undef_pts(mask_inner_undef,1), ...
            %         -sqrt(inner_undef_pts(mask_inner_undef,2).^2 + inner_undef_pts(mask_inner_undef,3).^2), ...
            %         3, 'filled', ...
            %         'MarkerFaceAlpha', alpha_t);

            scatter(outer_def_pts(mask_outer_def,1), ...
                    -sqrt(outer_def_pts(mask_outer_def,2).^2 + outer_def_pts(mask_outer_def,3).^2), ...
                    2, blueC,'filled', ...
                    'MarkerFaceAlpha', alpha_t);

            % scatter(outer_undef_pts(mask_outer_undef,1), ...
            %         -sqrt(outer_undef_pts(mask_outer_undef,2).^2 + outer_undef_pts(mask_outer_undef,3).^2), ...
            %         3, 'filled', ...
            %         'MarkerFaceAlpha', alpha_t);

            ylim([-0.4,-0.28])
            xlim([-0.15,0.15])
            daspect([1 1 1]);
            grid on
            xlabel('Y [m]');
            ylabel('Radii [m]');
            title(sprintf('Radial Cross Section at \\theta = %.0f°', theta0_deg(j)));
        end

        % -------------------------
        %  LONGITUDINAL CROSS SECTIONS
        % -------------------------
        for k = 1:length(long_x)
            x_slice = long_x(k);

            mask_inner_def   = abs(inner_def_pts(:,1) - x_slice) < tol & inner_def_pts(:,3) < -0.3;
            mask_inner_undef = abs(inner_undef_pts(:,1) - x_slice) < tol & inner_undef_pts(:,3) < -0.3;
            mask_outer_def   = abs(outer_def_pts(:,1) - x_slice) < tol & outer_def_pts(:,3) < -0.3;
            mask_outer_undef = abs(outer_undef_pts(:,1) - x_slice) < tol & outer_undef_pts(:,3) < -0.3;

            subplot(3,2,2*k-1)
            hold on

            scatter(inner_def_pts(mask_inner_def,2), inner_def_pts(mask_inner_def,3), ...
                    2, redC,'filled', 'MarkerFaceAlpha', alpha_t);
            
            % scatter(inner_undef_pts(mask_inner_undef,2), inner_undef_pts(mask_inner_undef,3), ...
            %         3, 'filled', 'MarkerFaceAlpha', alpha_t);

            scatter(outer_def_pts(mask_outer_def,2), outer_def_pts(mask_outer_def,3), ...
                    2, blueC,'filled', 'MarkerFaceAlpha', alpha_t);

            % scatter(outer_undef_pts(mask_outer_undef,2), outer_undef_pts(mask_outer_undef,3), ...
            %         3, 'filled', 'MarkerFaceAlpha', alpha_t);

            xlim([-0.22,0.22])
            daspect([1 1 1]);
            grid on
            xlabel('X [m]');
            ylabel('Y [m]');
            title(sprintf('Circumferential Cross Section at x = %.2f m', x_slice));
        end
    end
    sgtitle('Static Steering with 5\circ: Cross-Sections Over Time');
    axR = axes('Position',[0 0 1 1],'Visible','off'); % invisible axis
    colormap(axR, cmapR);
    
    cR = colorbar(axR, 'eastoutside');
    cR.Position = [0.9 0.1 0.015 0.8];  % below figure
    % cR.Label.String = 'Red metric';
    % cR.AxisLocation = 'out';
    cR.TickLabels = [];
    
    % ---- Blue colormap colorbar ----
    axB = axes('Position',[0 0 1 1],'Visible','off'); % invisible axis
    colormap(axB, cmapB);
    
    cB = colorbar(axB, 'eastoutside');
    cB.Position = [0.92 0.1 0.015 0.8]; % placed below the first one
    cB.Label.String = 'Deformation over time [s]';
    cB.AxisLocation = 'out';
    totalTime = length(data) * 0.067;    % total time in seconds
    clim(axB, [0 totalTime]);            % color axis in seconds
    
    % Choose tick positions along the colorbar in seconds
    numTicks = min(length(data), 6);     % max 6 ticks to avoid crowding
    tickPositions = linspace(0, totalTime, numTicks);
    
    cB.Ticks = tickPositions;            % positions along CLim
    cB.TickLabels = round(tickPositions, 2);
    
    % Set consistent limits
    % clim(axR, [1 length(data)]);
    % clim(axB, [0 N*0.067]);
    % Legend only once
    % lgd = legend({'Inner Def','Inner Undef','Outer Def','Outer Undef'});
    % lgd.Orientation = 'horizontal';
end

% function plot_cross_sections(data)
%     theta0_deg = [-80,-90,-100]*(pi/180);
%     long_x = [-0.05,0,0.05];
%     tol_deg = 0.1;
%     theta0 = deg2rad(theta0_deg);
%     tol = deg2rad(tol_deg);
%     figure;
% 
% 
%     for i = 1:length(data)
%         inner_def_pts = data{i}.pcd_inner_def.Location;
%         inner_undef_pts = data{i}.pcd_inner_undef.Location;
%         outer_def_pts = data{i}.pcd_outer_def.Location;
%         outer_undef_pts = data{i}.pcd_outer_undef.Location;
%         % Extract point positions
%         % pts = pc.Location;
%         % Y = pts(:,2);
%         % Z = pts(:,3);
% 
%         % Compute circumferential angle
%         % theta = atan2(Z, Y);      % in radians
%         theta_inner_def = atan2(inner_def_pts(:,3),inner_def_pts(:,2));
%         theta_inner_undef = atan2(inner_undef_pts(:,3),inner_undef_pts(:,2));
%         theta_outer_def = atan2(outer_def_pts(:,3),outer_def_pts(:,2));
%         theta_outer_undef = atan2(outer_undef_pts(:,3),outer_undef_pts(:,2));
%         % theta = unwrap(theta);    % avoid jump at +-pi
% 
%         % Mask near the target angle
%         for j = 1:length(theta0_deg)
%             theta0 = theta0_deg(j);
%             mask_inner_def = abs(theta_inner_def - theta0) < tol;
%             mask_inner_undef = abs(theta_inner_undef - theta0) < tol;
%             mask_outer_def = abs(theta_outer_def - theta0) < tol;
%             mask_outer_undef = abs(theta_outer_undef - theta0) < tol;
%             subplot(3,2,2*j)
%             hold on 
%             scatter(inner_def_pts(mask_inner_def,1), -sqrt(inner_def_pts(mask_inner_def,2).^2 + inner_def_pts(mask_inner_def,3).^2), 3, 'filled');
% 
%             scatter(inner_undef_pts(mask_inner_undef,1), -sqrt(inner_undef_pts(mask_inner_undef,2).^2 + inner_undef_pts(mask_inner_undef,3).^2), 3, 'filled');
%             scatter(outer_def_pts(mask_outer_def,1), -sqrt(outer_def_pts(mask_outer_def,2).^2 + outer_def_pts(mask_outer_def,3).^2), 3, 'filled');
%             scatter(outer_undef_pts(mask_outer_undef,1), -sqrt(outer_undef_pts(mask_outer_undef,2).^2 + outer_undef_pts(mask_outer_undef,3).^2), 3, 'filled');
%             hold off
%             ylim([-0.4,-0.28])
%             xlim([-0.15,0.15])
%             daspect([1 1 1]);         % 1:1 scaling between x and y
%             grid on
%             xlabel('Y [m]');
%             ylabel('Radii [m]');
%             title(sprintf('Radial Cross Section at \\theta = %.0f', rad2deg(theta0)));
%         end
% 
%         for k = 1:length(long_x)
%             z_slice = long_x(k);
%             mask_inner_def = abs(inner_def_pts(:,1) - z_slice) < tol & inner_def_pts(:,3) < -0.3;
%             mask_inner_undef = abs(inner_undef_pts(:,1) - z_slice) < tol & inner_undef_pts(:,3) < -0.3;
%             mask_outer_def = abs(outer_def_pts(:,1) - z_slice) < tol & outer_def_pts(:,3) < -0.3;
%             mask_outer_undef = abs(outer_undef_pts(:,1) - z_slice) < tol & outer_undef_pts(:,3) < -0.3;
% 
%             subplot(3,2,2*k-1)
%             hold on 
%             scatter(inner_def_pts(mask_inner_def,2),inner_def_pts(mask_inner_def,3), 3, 'filled',DisplayName='Inner Deformed Tyre')
% 
%             scatter(inner_undef_pts(mask_inner_undef,2),inner_undef_pts(mask_inner_undef,3), 3, 'filled',DisplayName='Inner Undeformed Tyre')
%             scatter(outer_def_pts(mask_outer_def,2),outer_def_pts(mask_outer_def,3), 3, 'filled',DisplayName='Outer Deformed Tyre')
%             scatter(outer_undef_pts(mask_outer_undef,2),outer_undef_pts(mask_outer_undef,3), 3, 'filled',DisplayName='Outer Undeformed Tyre')
%             hold off
%             xlim([-0.22,0.22])
%             daspect([1 1 1]);         % 1:1 scaling between x and y
%             grid on
%             xlabel('X [m]');
%             ylabel('Y [m]');
%             title(sprintf('Circumferencial Cross Section at x = %.2f m', z_slice));
%         end
%     end
%     lgd = legend();
%     lgd.Orientation = 'horizontal';
%     % ylim([-10,35])
%     % axis equal;
%     % xlabel('X [m]');
%     % ylabel('Radii [m]');
%     % title('Circumferential Angle vs Deformation');
%     % times_s = data_iter{1}.time(chooseIdx)/1000;
%     % legendStrings = arrayfun(@(t) sprintf('t = %.3f s', t), times_s, 'UniformOutput', false);
%     % legend(legendStrings)
%     % hold off;
% end

function overlay_contact_patch_m(data)
    figure;
    hold on;
    view(3);
    grid on;
    
    N = numel(data);
    
    % Large colour pool (increase if needed)
    maxPossibleLugs = 100;
    baseColors = lines(maxPossibleLugs);
    
    % Tracking variables
    prevCentroids = [];
    prevIDs = [];
    nextID = 1;
    
    matchTolerance = 0.02;   % <-- tune based on your tyre scale
    
    for t = 1:N
    
        % Get contact patch points
        pts = data{t}.pcd_contact_patch.Location;
        allPts = pts(:,1:2);
    
        % Extract lug boundaries
        [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
            'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);
    
        nLugs = numel(boundaries);
    
        if nLugs == 0
            prevCentroids = [];
            prevIDs = [];
            continue;
        end
    
        % Compute centroids
        centroids = zeros(nLugs,2);
        for k = 1:nLugs
            poly = boundaries{k};
            centroids(k,:) = mean(poly(:,1:2),1);
        end
    
        % Assign persistent IDs
        currentIDs = zeros(nLugs,1);
    
        if isempty(prevCentroids)
            % First frame → assign new IDs
            currentIDs = (nextID:nextID+nLugs-1)';
            nextID = nextID + nLugs;
    
        else
            for k = 1:nLugs
    
                dists = vecnorm(prevCentroids - centroids(k,:), 2, 2);
    
                [minDist, idx] = min(dists);
    
                if minDist < matchTolerance
                    currentIDs(k) = prevIDs(idx);
                else
                    currentIDs(k) = nextID;
                    nextID = nextID + 1;
                end
    
            end
        end
    
        % Plot lugs with persistent colour
        for k = 1:nLugs
    
            poly = boundaries{k};
    
            lugID = currentIDs(k);
    
            % Protect against exceeding colour pool
            colorIndex = mod(lugID-1, size(baseColors,1)) + 1;
            c = baseColors(colorIndex,:);
    
            % Optional time brightness modulation
            alphaTime = t / N;
            cMod = (1-alphaTime)*c + alphaTime*[1 1 1];
    
            plot3(poly(:,1), ...
                  poly(:,2), ...
                  (t/250)*ones(size(poly,1),1), ...
                  '-', ...
                  'LineWidth',1.6, ...
                  'Color',cMod);
        end
    
        % Update tracking memory
        prevCentroids = centroids;
        prevIDs = currentIDs;
    
    end
    
    xlabel('X');
    ylabel('Y');
    zlabel('Time');
    title('Persistent Lug Tracking');
end

function overlay_contact_patch(data)
    N = length(data);
    figure;
    hold on;
    xlabel('X'); ylabel('Y'); zlabel('Time step');
    title('Detected lug boundaries over time');
    axis equal;

    % ptsFull = data{1}.pcd_outer_def.Location;     % full cloud
    % mask = data{1}.contact_patch_mask;            % logical mask
    % 
    % contactPts = ptsFull(mask,:);
    % contactIdx = find(mask);   % global indices of contact points
    % 
    % xy = double(contactPts(:,1:2));
    % 
    % shrinkFactor = 1;  % tune (0 = convex hull, 1 = tight boundary)
    % k = boundary(xy(:,1), xy(:,2), shrinkFactor);
    % 
    % boundaryIdx = contactIdx(k);
    maxLugs = 0;
    % First pass: find max number of lugs to create colormap
    for t = 1:N
        pts = data{t}.pcd_contact_patch.Location;
        allPts = [pts(:,1), pts(:,2)];
        [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
            'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);
        maxLugs = max(maxLugs, numel(boundaries));
    end

    baseColors = lines(maxLugs);  % base colors for lugs

    % sampleIdx = (200000:200:201000);
    % nSamples = numel(sampleIdx);
    % 
    % Global fixed sampling of full cloud
    sampleIdx = 1:100:407040; %[1:100:150000 350000:100:407040];
    
    % Contact patch mask at reference time (e.g., t = 1)
    refMask = data{1}.contact_patch_mask;
    
    % Only keep sampled points that are in contact at t=1
    trackIdx = sampleIdx(refMask(sampleIdx)); %boundaryIdx; 
    
    nTrack = numel(trackIdx);
    
    trajX = zeros(nTrack, N);
    trajY = zeros(nTrack, N);
    trajZ = zeros(nTrack, N);

    baseColors = lines(50);  % large enough pool
    lugTracks = struct();    % will store persistent IDs
    nextID = 1;
    prevCentroids = [];
    prevIDs = [];
    for t = 1:N
        pts = data{t}.pcd_contact_patch.Location;
        allPts = [pts(:,1), pts(:,2)];

        [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
            'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);

        nLugs = numel(boundaries);
        centroids = zeros(nLugs,2);

        for k = 1:nLugs
            poly = boundaries{k};
            centroids(k,:) = mean(poly(:,1:2),1);
        end

        nLugs = numel(boundaries);

        for k = 1:nLugs
            poly = boundaries{k};
            % Modulate color by time step: e.g., interpolate to white
            c = baseColors(k,:);
            alphaTime = t / N; % 0->early, 1->late
            % Blend with white to make later timesteps brighter
            cMod = (1-alphaTime)*c + alphaTime*[1 1 1]*0.5;

            plot3(poly(:,1), poly(:,2), t*ones(size(poly,1),1)/250, ...
                '-', 'LineWidth', 2, 'Color', cMod);
        end

        % Scatter points lightly
        scatter3(allPts(:,1), allPts(:,2), t*ones(size(allPts,1),1)/250, ...
            8, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha',0.1);

        % plot3(allPts((1:500:end),1), allPts((1:500:end),2), t*ones(size(allPts((1:500:end),:),1),1)/250);
    end
    % % % % cMap = parula(N);      % N time steps
    % % % % for t = 1:N
    % % % %     pts = data{t}.pcd_contact_patch.Location;
    % % % %     allPts = [pts(:,1), pts(:,2)];
    % % % % 
    % % % %     % pts = data{t}.pcd_contact_patch.Location;
    % % % %     outerPts = data{t}.pcd_outer_def.Location;
    % % % %     mask = data{t}.contact_patch_mask;
    % % % %     ptsTracked = outerPts(trackIdx,:);
    % % % %     inContactNow = mask(trackIdx);
    % % % % 
    % % % %     trajX(inContactNow,t) = ptsTracked(inContactNow,1);
    % % % %     trajY(inContactNow,t) = ptsTracked(inContactNow,2);
    % % % %     trajZ(inContactNow,t) = t/250;
    % % % % 
    % % % %     trajX(~inContactNow,t) = NaN;
    % % % %     trajY(~inContactNow,t) = NaN;
    % % % %     trajZ(~inContactNow,t) = NaN;
    % % % % 
    % % % %     [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
    % % % %         'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);
    % % % % 
    % % % %     nLugs = numel(boundaries);
    % % % % 
    % % % %     cMod = cMap(t,:);      % color for this time step
    % % % %     if t==1 || t==N%mod(t,4) == 0
    % % % %         for k = 1:nLugs
    % % % %             poly = boundaries{k};
    % % % %             % Modulate color by time step: e.g., interpolate to white
    % % % %             c = baseColors(k,:);
    % % % % 
    % % % %             alphaTime = t / N; % 0->early, 1->late
    % % % %             % Blend with white to make later timesteps brighter
    % % % %             cMod = c .* cMap(t,:);
    % % % %             % cMod = (1-alphaTime)*cMod + alphaTime*[0.5 0.5 0.5];
    % % % % 
    % % % % 
    % % % %             plot3(poly(:,1), poly(:,2), t*ones(size(poly,1),1)/250, ...
    % % % %                 '-', 'LineWidth', 1.6, 'Color', cMod); %[cMod, 0.3]
    % % % %         end
    % % % %     end
    % % % %     % Scatter points lightly
    % % % %     % scatter3(allPts(:,1), allPts(:,2), t*ones(size(allPts,1),1)/250, ...
    % % % %     %     8, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha',0.1);
    % % % % 
    % % % %     % plot3(allPts((1:500:end),1), allPts((1:500:end),2), t*ones(size(allPts((1:500:end),:),1),1)/250);
    % % % % end
    % % % % hold on
    % % % % for i = 1:nTrack
    % % % %     plot3(trajX(i,:), trajY(i,:), trajZ(i,:), '-s','LineWidth',1.2,'MarkerSize',2);
    % % % % end
    view(3); % 3D view
    grid on;
    hold off;
end

function overlay_contact_patch_t(data)
    N = length(data);
    figure;
    hold on;
    xlabel('X'); ylabel('Y'); zlabel('Time step');
    title('Detected lug boundaries over time');
    axis equal;

    % ptsFull = data{1}.pcd_outer_def.Location;     % full cloud
    % mask = data{1}.contact_patch_mask;            % logical mask
    % 
    % contactPts = ptsFull(mask,:);
    % contactIdx = find(mask);   % global indices of contact points
    % 
    % xy = double(contactPts(:,1:2));
    % 
    % shrinkFactor = 1;  % tune (0 = convex hull, 1 = tight boundary)
    % k = boundary(xy(:,1), xy(:,2), shrinkFactor);
    % 
    % boundaryIdx = contactIdx(k);
    maxLugs = 0;
    % First pass: find max number of lugs to create colormap
    for t = 1:N
        pts = data{t}.pcd_contact_patch.Location;
        allPts = [pts(:,1), pts(:,2)];
        [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
            'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);
        maxLugs = max(maxLugs, numel(boundaries));
    end

    baseColors = lines(maxLugs);  % base colors for lugs

    % sampleIdx = (200000:200:201000);
    % nSamples = numel(sampleIdx);
    % 
    % Global fixed sampling of full cloud
    sampleIdx = 1:2:407040; %[1:100:150000 350000:100:407040];
    
    % Contact patch mask at reference time (e.g., t = 1)
    refMask = data{1}.contact_patch_mask;
    
    % Only keep sampled points that are in contact at t=1
    trackIdx = sampleIdx(refMask(sampleIdx)); %boundaryIdx; 
    
    nTrack = numel(trackIdx);
    
    trajX = zeros(nTrack, N);
    trajY = zeros(nTrack, N);
    trajZ = zeros(nTrack, N);

    baseColors = lines(50);  % large enough pool
    lugTracks = struct();    % will store persistent IDs
    nextID = 1;
    prevCentroids = [];
    prevIDs = [];
    % % % for t = 1:N
    % % %     pts = data{t}.pcd_contact_patch.Location;
    % % %     allPts = [pts(:,1), pts(:,2)];
    % % % 
    % % %     [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
    % % %         'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);
    % % % 
    % % %     nLugs = numel(boundaries);
    % % %     centroids = zeros(nLugs,2);
    % % % 
    % % %     for k = 1:nLugs
    % % %         poly = boundaries{k};
    % % %         centroids(k,:) = mean(poly(:,1:2),1);
    % % %     end
    % % % 
    % % %     nLugs = numel(boundaries);
    % % % 
    % % %     for k = 1:nLugs
    % % %         poly = boundaries{k};
    % % %         % Modulate color by time step: e.g., interpolate to white
    % % %         c = baseColors(k,:);
    % % %         alphaTime = t / N; % 0->early, 1->late
    % % %         % Blend with white to make later timesteps brighter
    % % %         cMod = (1-alphaTime)*c + alphaTime*[1 1 1]*0.5;
    % % % 
    % % %         plot3(poly(:,1), poly(:,2), t*ones(size(poly,1),1)/250, ...
    % % %             '-', 'LineWidth', 2, 'Color', cMod);
    % % %     end
    % % % 
    % % %     % Scatter points lightly
    % % %     scatter3(allPts(:,1), allPts(:,2), t*ones(size(allPts,1),1)/250, ...
    % % %         8, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha',0.1);
    % % % 
    % % %     % plot3(allPts((1:500:end),1), allPts((1:500:end),2), t*ones(size(allPts((1:500:end),:),1),1)/250);
    % % % end
    cMap = parula(N);      % N time steps
    for t = 1:N
        pts = data{t}.pcd_contact_patch.Location;
        allPts = [pts(:,1), pts(:,2)];

        % pts = data{t}.pcd_contact_patch.Location;
        outerPts = data{t}.pcd_outer_def.Location;
        mask = data{t}.contact_patch_mask;
        ptsTracked = outerPts(trackIdx,:);
        inContactNow = mask(trackIdx);

        trajX(inContactNow,t) = ptsTracked(inContactNow,1);
        trajY(inContactNow,t) = ptsTracked(inContactNow,2);
        trajZ(inContactNow,t) = t/250;

        trajX(~inContactNow,t) = NaN;
        trajY(~inContactNow,t) = NaN;
        trajZ(~inContactNow,t) = NaN;

        [boundaries, ~, ~] = find_lug_boundaries(allPts, ...
            'eps',0.008, 'MinPts',30, 'Alpha',0.04, 'Plot',false);

        nLugs = numel(boundaries);

        cMod = cMap(t,:);      % color for this time step
        if t==1 || t==N%mod(t,4) == 0
            for k = 1:nLugs
                poly = boundaries{k};
                % Modulate color by time step: e.g., interpolate to white
                c = baseColors(k,:);

                alphaTime = t / N; % 0->early, 1->late
                % Blend with white to make later timesteps brighter
                cMod = c .* cMap(t,:);
                % cMod = (1-alphaTime)*cMod + alphaTime*[0.5 0.5 0.5];


                plot3(poly(:,1), poly(:,2), t*ones(size(poly,1),1)/250, ...
                    '-', 'LineWidth', 1.6, 'Color', cMod); %[cMod, 0.3]
            end
        end
        % Scatter points lightly
        % scatter3(allPts(:,1), allPts(:,2), t*ones(size(allPts,1),1)/250, ...
        %     8, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha',0.1);

        % plot3(allPts((1:500:end),1), allPts((1:500:end),2), t*ones(size(allPts((1:500:end),:),1),1)/250);
    end
    hold on
    nx = 5;   % number of bins in x
    ny = 10;   % number of bins in y
    
    % xEdges = linspace(min(trajX(:,1)), max(trajX(:,1)), nx+1);
    % yEdges = linspace(min(trajY(:,1)), max(trajY(:,1)), ny+1);
    xEdges = linspace(-0.12, 0, nx+1);
    yEdges = linspace(-0.08, 0.05, ny+1);
    selectedIdx = [];

    for ix = 1:nx
        for iy = 1:ny
            
            inCell = trajX(:,1) >= xEdges(ix) & trajX(:,1) < xEdges(ix+1) & ...
                     trajY(:,1) >= yEdges(iy) & trajY(:,1) < yEdges(iy+1);
            
            candidates = find(inCell);
            
            if ~isempty(candidates)
                selectedIdx(end+1) = candidates(1); % or random
            end
        end
    end
    for i = selectedIdx %1:20:nTrack
        plot3(trajX(i,:), trajY(i,:), trajZ(i,:), '-s','LineWidth',1.2,'MarkerSize',2);
    end
    view(3); % 3D view
    grid on;
    hold off;
end

function plot_time_series_def(data)

    N = length(data);

    % Choose a fixed lateral grid (Y-axis)
    y_min = -0.18;
    y_max =  0.18;
    ny    = 1500;
    y_grid = linspace(y_min, y_max, ny);

    % Prepare Z(t,y)
    Z = zeros(N, ny);
    for t = N:-1:1
        pts = data{t}.pcd_outer_def.Location;

        % Extract cross-section slice (longitudinal)
        % Z = vertical coord, Y = lateral coord
        mask = abs(pts(:,1)+0.05) < 0.0005 & abs(pts(:,2)) < 0.2 & pts(:,3) < -0.3 & pts(:,3) > -0.37; 
        Y = pts(mask,2);
        Zvals = pts(mask,3);
        [Y_unique, ~, idx] = unique(Y);
        Z_unique = accumarray(idx, Zvals, [], @mean);   % average Z for same Y

        % Interpolate onto uniform grid so waterfall works
        Z(t,:) = interp1(Y_unique, Z_unique, y_grid, 'linear', 'extrap');
    end

    % Plot
    figure; 
    waterfall(y_grid, (1:N)/500, -Z);
    xlabel('Longitudinal coordinate X (m)');
    ylabel('Time step');
    zlabel('Vertical coordinates Z (m)');
    % zlim([-0.42,-0.3])
    zlim([0.3,0.42])
    daspect([1 1 1]);
    title('Longitudinal Cross-Section Waterfall Over Time');

    N = length(data);

    % Choose a fixed longitudinal grid (X-axis)
    x_min = -0.118;
    x_max =  0.13;
    nx    = 500;
    x_grid = linspace(x_min, x_max, nx);

    % Prepare Z(t,y)
    Z = zeros(N, ny);
    for t = N:-1:1
        pts = data{t}.pcd_outer_def.Location;

        % Extract cross-section slice (longitudinal)
        % Z = vertical coord, Y = lateral coord
        mask = abs(pts(:,2)) < 0.0005 & abs(pts(:,1)) < 0.2 & pts(:,3) < -0.3 & pts(:,3) > -0.37; 
        X = pts(mask,1);
        Zvals = pts(mask,3);
        [X_unique, ~, idx] = unique(X);
        Z_unique = accumarray(idx, Zvals, [], @mean);   % average Z for same Y

        % Interpolate onto uniform grid so waterfall works
        Z(t,:) = interp1(X_unique, Z_unique, x_grid, 'linear', 'extrap');
    end

    % Plot
    figure; 
    waterfall(x_grid, (1:N)/500, -Z);
    xlabel('Lateral coordinate Y (m)');
    ylabel('Time step');
    zlabel('Vertical coordinates Z (m)');
    xlim([-0.118,0.13])
    zlim([0.3,0.42])
    % xlim([-0.12,0.139])
    daspect([1 1 1]);
    title('Lateral Cross-Section Waterfall Over Time');


end

function plot_hist_time(data)
    N = length(data);
    %figure;
    % for i = N:-1:1
    %     pts = data{i}.pcd_inner_def.Location;
    %     mask = pts(:,3) < -0.28;
    %     histogram(data{i}.dist_inner_def(mask),1000,FaceAlpha=1,EdgeAlpha=0.03)
    %     hold on
    % end
    
    % for i = 1:N
    %     pts = data{i}.pcd_inner_def.Location;
    %     mask = pts(:,3) < -0.28;
    %     [counts, edges] = histcounts(data{i}.dist_inner_def(mask), 10000);
    %     binCenters = edges(1:end-1) + diff(edges)/2;
    %     plot(binCenters, counts);
    %     hold on
    % end

    pts = data{1}.pcd_inner_def.Location;
    mask = pts(:,3) < -0.28;
    [counts, edges] = histcounts(data{1}.dist_inner_def(mask), 10000);
    % binCenters = edges(1:end-1) + diff(edges)/2;
    %     h = histogram(data{1}.dist_inner_def(mask), 10000);
    % 
    %     h.DataTipTemplate.DataTipRows(1).Label = 'Bin center';
    %     h.DataTipTemplate.DataTipRows(1).Value = (h.BinEdges(1:end-1) + [0 diff(h.BinEdges)/2]);
    % 
    % h.DataTipTemplate.DataTipRows(1).Format = '%.3f';

    nbins = 300;

    edges = linspace(-0.05, 0.01, nbins+1);
    centers = edges(1:end-1) + diff(edges)/2;
    
    Z = zeros(N, nbins);
    
    for t = 1:N
        pts = data{t}.pcd_inner_def.Location;
        mask = pts(:,3) < -0.28;   
        Z(t,:) = histcounts(data{t}.dist_inner_def(mask), edges); %, 'Normalization','pdf'
    end
    
    figure;
    waterfall(centers, 1:N, Z);
    xlabel('Deformation [m]');
    ylabel('Time');
    yticks([])
    zlabel('Number of Points');
    title('Deformation Histogram of STTR Cleat over time');
    % 
    % figure;
    % pts = data{1}.pcd_inner_def.Location;
    % mask = pts(:,3) < -0.28;  
    % histogram(data{1}.dist_inner_def(mask), edges)
    % hold on
    % pts = data{end-5}.pcd_inner_def.Location;
    % mask = pts(:,3) < -0.28;  
    % histogram(data{end-5}.dist_inner_def(mask), edges, FaceColor=[1 0 0])
    % 
    % % pts = data_s{1}.pcd_inner_def.Location;
    % % mask = pts(:,3) < -0.28;  
    % % histogram(data_s{1}.dist_inner_def(mask), edges)
    % 
    % pts = data_s{end}.pcd_inner_def.Location;
    % mask = pts(:,3) < -0.28;  
    % histogram(data_s{end}.dist_inner_def(mask), edges, FaceColor=[0 1 0])
    % hold off
end


function track_rolling_points(data)
    N = length(data);
    disp_fields_unmasked = cell(N-1,1);
    disp_fields = cell(N-1,1);
    vel_est = zeros(N-1,1);

    % Extract time vector (same for all entries)
    dt = data{1}.dt/1000;          % dt is length N-1
    time_vec = data{1}.time/1000;  % N×1 time vector

    row = 1;%480/2;
    column = 848/2;
    traj = zeros(N-1, 3);
    u = zeros(N-1, 3);
    d_traj = zeros(N-1, 3);
    v_traj = zeros(N-1, 3);
    deform_x = zeros(N-1,1);
    deform_y = zeros(N-1,1);
    deform_z = zeros(N-1,1);
    idx = 848*row+column;
    undef = zeros(N-1, 3);
    def = zeros(N-1, 3);

    for i = 2:N
        % pc = data{i}.pcd_inner_def;
        pts_curr = data{i}.pcd_inner_def.Location; %(data{i}.curr_valid_mask & data{i}.prev_valid_mask,:); %& data{i}.contact_patch_mask
        pts_prev = data{i-1}.pcd_inner_def.Location; %(data{i}.curr_valid_mask & data{i}.prev_valid_mask,:); %& data{i}.contact_patch_mask
        % u(i-1,:) = data{i-1}.dist_3d(idx,:);
        undef(i-1,:) = data{i-1}.pcd_inner_undef.Location(idx,:);
        def(i-1,:) = data{i-1}.pcd_inner_def.Location(idx,:);
        traj(i-1,:) = pts_prev(idx,:);
        disp_fields{i-1} = pts_curr - pts_prev;                   % Nx3 array
        d_traj(i-1,:) = disp_fields{i-1}(idx,:);  
        v_traj(i-1,:) = disp_fields{i-1}(idx,:)./(time_vec(i-1));
    end
    R = sqrt(undef(:,2).^2 + undef(:,3).^2);  % approximate radius

    theta_undef = unwrap(atan2(undef(:,3), undef(:,2)));
    theta_def   = unwrap(atan2(def(:,3), def(:,2)));

    s_undef = R .* theta_undef;
    s_def   = R .* theta_def;

    t_hat = [-sin(theta_undef), cos(theta_undef)]; % tangential unit vector in Y-Z
    v_theta = v_traj(:,2) .* t_hat(:,1) + v_traj(:,3) .* t_hat(:,2);  % dot product
    dcs=v_theta' .* (dt(110:205)+1e-12);
    s_cum = cumsum(v_theta(3:end)' .* dt(112:205))';
    disp(size(dcs))
    disp([v_theta',size(dt(110:205))])
    disp(s_cum)


    u_theta = s_def - s_undef;   % circumferential deformation
    strain_theta = u_theta./s_undef;
    disp(size(strain_theta))

    % undef = data{30}.pcd_inner_undef.Location;
    % def = data{30}.pcd_inner_def.Location;
    % 
    % mask_contact = undef(:,1) > -0.001 & undef(:,1) < 0.001 & ...  % X range
    %            undef(:,2) > -0.03 & undef(:,2) < 0.03 & ...    % Y range
    %            undef(:,3) > -0.42 & undef(:,3) < -0.3;      % Z range
    % R = sqrt(undef(:,2).^2 + undef(:,3).^2);  % approximate radius
    % 
    % theta_undef = unwrap(atan2(undef(:,1), undef(:,3)));
    % theta_def   = unwrap(atan2(def(:,1), def(:,3)));
    % 
    % % s_undef = R .* theta_undef;
    % % s_def   = R .* theta_def;
    % 
    % s_undef = undef(:,2);  % longitudinal coordinate along contact patch
    % s_def   = def(:,2);
    % 
    % u_theta = s_def(mask_contact) - s_undef(mask_contact);  % delta along rolling direction
    % 
    % strain_theta = u_theta; %./ s_undef(mask_contact); %s_undef(mask_contact);  % simple engineering strain
    % 
    % [s_sorted, idx] = sort(s_undef(mask_contact));
    % strain_sorted = movmean(strain_theta(idx),7);
    % 
    % figure;
    % plot(s_sorted, strain_sorted, 'LineWidth',1.5);
    % xlabel('Position along contact patch [m]');
    % ylabel('Circumferential strain');
    % grid on;
    % title('Circumferential strain profile along contact patch');

    figure; 
    scatter3(traj(:,1), traj(:,2), traj(:,3), ...
        10, 'filled');
    grid on; axis equal;
    hold on;
    % scatter3(d_traj(:,1), d_traj(:,2), d_traj(:,3),...
    %     1, 'k','filled');
    scatter3(pts_prev(:,1), pts_prev(:,2), pts_prev(:,3), ...
        1, 'k','filled');
    hold off;
    xlabel("X [m]")
    ylabel("Y [m]")
    zlabel("Z [m]")

    figure; 
    disp(size(d_traj))
    disp(size(time_vec(110:210)))
    % scatter(traj(:,2),deform)
    % plot(time_vec(110:205),v_traj(:,1))
    % figure; grid on; axis equal;
    % plot(time_vec(110:205),v_traj(:,2))
    % figure; grid on; axis equal;
    % plot(time_vec(110:205),v_traj(:,3))
    % plot(time_vec(110:204),strain(:,1))
    % figure; grid on; axis equal;
    % plot(time_vec(110:204),strain(:,2))
    % figure; grid on; axis equal;
    % plot(time_vec(110:204),strain(:,3))
    % plot(time_vec(110:205),u(:,1))
    % figure; grid on; axis equal;
    % plot(time_vec(110:205),u(:,2))
    % figure; grid on; axis equal;
    % plot(time_vec(110:205),u(:,3))
    plot(time_vec(115:205),strain_theta(6:end))
    xlabel("Time [s]")
    ylabel("Strain_{\theta}")
    title("Longitudinal Strain of Tracked Point over time")
    grid on;
    figure; 
    plot(s_cum(5:end),strain_theta(7:end))
    % xlabel("Time [s]")
    % ylabel("Strain_{\theta}")
    % title("Longitudinal Strain of Tracked Point over time")
    grid on;
    figure; 
    plot(rad2deg(theta_undef(5:end)),movmean(strain_theta(5:end),10))
    xlabel("Degrees \degrees")
    ylabel("Strain_{\theta}")
    title("Longitudinal Strain of Tracked Point over Arc")
    grid on;
    figure; 
    plot(time_vec(110:205),u_theta)
    xlabel("Time [s]")
    ylabel("Deformation_{\theta}")
    title("Longitudinal Deformation of Tracked Point over time")
    grid on;
end

function vel_est = plot_vel(data)

    N = length(data);
    disp_fields_unmasked = cell(N-1,1);
    disp_fields = cell(N-1,1);
    vel_fields = cell(N-1,1);
    vel_est = zeros(N-1,1);

    % Extract time vector (same for all entries)
    dt = data{1}.dt;          % dt is length N-1
    time_vec = data{1}.time;  % N×1 time vector

    for i = 2:N
        pts_curr = data{i}.pcd_outer_def.Location(data{i}.curr_valid_mask & data{i}.prev_valid_mask ,:); %& data{i}.contact_patch_mask
        pts_prev = data{i-1}.pcd_outer_def.Location(data{i}.curr_valid_mask & data{i}.prev_valid_mask,:); %& data{i}.contact_patch_mask

        % If number of points mismatch, stop
        if size(pts_curr,1) ~= size(pts_prev,1)
            error("Mismatched point counts at frame %d and %d", i-1, i);
        end

        disp_fields_unmasked{i-1} = pts_curr - pts_prev;
        % disp(size(disp_fields_unmasked{i-1}))
        % disp(size(data{i}.curr_valid_mask))
        % disp(size(pts_curr))% Nx3 displacement
        location_mask = (pts_curr(:,1) > -0.025) & (pts_curr(:,1) < 0.025) & (pts_curr(:,2) > -0.025) & (pts_curr(:,2) < 0.025) & (pts_curr(:,3) > -0.42) & (pts_curr(:,3) < -0.35);
        disp_fields{i-1} = disp_fields_unmasked{i-1}(location_mask,:);%(data{i-1}.curr_valid_mask & data{i-1}.prev_valid_mask & data{i-1}.contact_patch_mask); %data.contact_patch_mask
        vel_fields{i-1} = disp_fields{i-1}./(dt(i)/1000);
        % disp(size(vel_fields{i-1}))
        vel_mag = vel_fields{i-1}(:,3); %vecnorm(vel_fields{i-1}, 2, 2);  % Nx1 vector of Euclidean norms 
        % disp(vel_fields{i-1}(:,3))
        


        % Compute 95th percentile threshold
        thresh_high = prctile(vel_mag, 92);
        thresh_low = prctile(vel_mag, 90);
        % disp([thresh_low,thresh_high])
        
        % Select velocities above threshold
        vel_high = vel_mag; %(vel_mag >= thresh_low & vel_mag <= thresh_high);
        % disp(vel_high)
        % figure(i);
        % histogram(vel_high,100)
        % Take mean of those
        vel_est(i-1) = mean(vel_high);
        % vel_est(i-1) = mean(percentile(vecnorm(vel_fields{i-1}, 2, 2),95));
        % disp(vel_est(i-1))
    end

    % Plot velocity estimate vs time
    figure;
    % disp(length(time_vec));
    % disp(length(vel_est));
    plot(vel_est(1:end),'LineWidth',1.5)
    xlabel("Time (s)")
    ylabel("Velocity (m/s)")
    title("Max Tyre Contact Velocity Response to STTR Plate")
    grid on
end

function extract_frames_from_mp4(videoPath, frameIdxList, outFolder)

    if ~exist(outFolder,'dir')
        mkdir(outFolder);
    end

    v = VideoReader(videoPath);
    sampleHz = 15;
    dt_sample = 1/sampleHz;
    for k = 1:length(frameIdxList)
        sampleIdx = frameIdxList(k);
    
        % Convert sample index -> timestamp
        t = (sampleIdx - 1) * dt_sample;
    
        % Clamp to video duration
        t = min(t, v.Duration - 0.001);
    
        % Jump to correct timestamp
        v.CurrentTime = t;
    
        % Read frame closest to this time
        frame = readFrame(v);
    
        % Save output
        outName = fullfile(outFolder, sprintf('sample_%04d.png', sampleIdx));
        imwrite(frame, outName);
        fprintf('Saved frame %d → %s\n', sampleIdx, videoPath);
    end
end

function plot_one_data(data)
    for j = 1:length(data)
        pts = data{j}.pcd_inner_def.Location;
        mask = pts(:,3) < -0.28;
        % disp(size())
         % & data{j}.contact_patch_mask'
        pts = pts(mask,:);
        Smin = -0.035;  
        Smax = 0.01;
    
        cmap = slanCM('plasma',256);
    
        Sclip = min(max(data{j}.dist_inner_def(mask), Smin), Smax);
        S_norm = (Sclip - Smin) / (Smax - Smin);
        idx = min(max(round(S_norm*255)+1,1),256);
        col = cmap(idx,:);
    
        fig = figure('Position',[100 100 900 600]);
        
        % hScatter = scatter3(pts(:,1), pts(:,2), pts(:,3), ...
        %     10, col, 'filled');
        hScatter = scatter3(pts(:,1), pts(:,2), pts(:,3), ...
            10, data{j}.dist_inner_def(mask) , 'filled');
        axis('equal')
        xlim([-0.15,0.15])
        ylim([-0.22,0.22])
        zlim([-0.4,-0.28])
        hold('on');
        
        xlabel('X'), ylabel('Y'), zlabel('Z')
        % title(ax1,'3D Point Cloud')
        colormap(cmap)
        clim([Smin Smax])
        % cb = colorbar(Location','eastoutside');
        % % cb.Label.String = 'Signed deformation';
        % cb.FontSize = 11;
        % cb.RulerLocation = 'left';
    
        hold('off')
    
        % outname = sprintf('iter_%03d_inner.png', j);
        % % outname = sprintf('cleat_6000_inner.png', j);
        % exportgraphics(fig, outname, 'Resolution', 300);

        % optional: close the figure to avoid memory buildup
        %close(fig);
    end
end


function plot_pre_data(data)
    for j = 1:length(data)
        pts = data{j}.pcd_inner_def.Location;
        mask = pts(:,3) < -0.28;
        pts = pts(mask,:);
        Smin = -0.035;  
        Smax = 0.01;
    
        cmap = slanCM('plasma',256);
    
        Sclip = min(max(data{j}.dist_inner_def(mask), Smin), Smax);
        S_norm = (Sclip - Smin) / (Smax - Smin);
        idx = min(max(round(S_norm*255)+1,1),256);
        col = cmap(idx,:);
    
        fig = figure('Position',[100 100 900 600]);
        
        ax1 = subplot(1,2,1);
    
        
        hScatter = scatter3(ax1, pts(:,1), pts(:,2), pts(:,3), ...
            10, col, 'filled');
        axis(ax1,'equal')
        xlim(ax1, [-0.15,0.15])
        ylim(ax1, [-0.22,0.22])
        zlim(ax1, [-0.4,-0.28])
        hold(ax1,'on');
    
        
        xlabel(ax1,'X'), ylabel(ax1,'Y'), zlabel(ax1,'Z')
        % title(ax1,'3D Point Cloud')
        colormap(ax1, cmap)
        clim(ax1,[Smin Smax])
        cb = colorbar(ax1,'Location','eastoutside');
        % cb.Label.String = 'Signed deformation';
        cb.FontSize = 11;
        cb.RulerLocation = 'left';
    
        hold(ax1,'off')
    
        ax3 = subplot(1,2,2);
    
        nbins = 1000;
        [counts, edges] = histcounts(data{j}.dist_inner_def(mask), nbins);
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
        xlim(ax3,[0,10000])
        ylim(ax3,[Smin Smax])
        title(ax3,'Signed Deformation','FontSize',11)
        ax3.YTickLabel = [];
        ax1.Position = [0.05 0.1 0.6 0.8];
        ax3.Position = [0.724 0.1 0.1 0.8];

        % outname = sprintf('iter_%03d_outer.png', j);
        % exportgraphics(fig, outname, 'Resolution', 300);
        % 
        % % optional: close the figure to avoid memory buildup
        % close(fig);
    end
end

function input_data = load_data(chooseIdx, files)

    % Preallocate cell array
    input_data = cell(numel(chooseIdx), 1);

    for k = 1:numel(chooseIdx)
        currentIdx = chooseIdx(k);

        % ----- Load data -----
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
        % dist_3d = data.dist_3d;

        % ----- Filtering masks -----
        mask_inner_def = inner_def(:,3) > 0.07;

        % ----- Convert to pointCloud -----
        pcd_inner_def = pointCloud(inner_def);
        pcd_inner_undef = pointCloud(inner_undef);
        pcd_outer_def = pointCloud(outer_def);
        pcd_outer_undef = pointCloud(outer_undef);
        pcd_tread_def = pointCloud(tread_def);
        pcd_tread_undef = pointCloud(tread_undef);
        pcd_contact_patch = pointCloud(contact_patch);

        % ----- Rotate to camera view -----
        pcd_inner_def = rotate_cam_view(pcd_inner_def);
        pcd_inner_undef = rotate_cam_view(pcd_inner_undef);
        pcd_outer_def = rotate_cam_view(pcd_outer_def);
        pcd_outer_undef = rotate_cam_view(pcd_outer_undef);
        pcd_tread_def = rotate_cam_view(pcd_tread_def);
        pcd_tread_undef = rotate_cam_view(pcd_tread_undef);
        pcd_contact_patch = rotate_cam_view(pcd_contact_patch);

        % ----- Store in struct -----
        S.dt = dt;
        S.time = time;
        S.pcd_inner_def = pcd_inner_def;
        S.pcd_inner_undef = pcd_inner_undef;
        S.pcd_outer_def = pcd_outer_def;
        S.pcd_outer_undef = pcd_outer_undef;
        S.pcd_tread_def = pcd_tread_def;
        S.pcd_tread_undef = pcd_tread_undef;
        S.pcd_contact_patch = pcd_contact_patch;
        S.dist_inner_def = dist_inner_def;
        S.dist_inner_to_outer_undef = dist_inner_to_outer_undef;
        S.dist_inner_to_tread_undef = dist_inner_to_tread_undef;
        S.curr_valid_mask = curr_valid_mask;
        S.prev_valid_mask = prev_valid_mask;
        S.contact_patch_mask = contact_patch_mask;

        % S.dist_3d = dist_3d;

        % ----- Save into cell array -----
        input_data{k} = S;
    end
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
