clear all

images = {imread('000000.jpg'), imread('000170.jpg'), imread('000190.jpg'), imread('000210.jpg')};
% overlay_images(images);
% overlay_color_tinted(images);
% overlay_colorcoded_images(images,linspace(0.1, 0.2, numel(images)));
overlay_edge_progression(images);

function overlay_edge_progression(images)
% images = cell array of grayscale images (e.g., {img1, img2, img3, ...})

    N = length(images);

    % --- Choose distinct colors automatically ---
    cmap = lines(N);     % or hsv(N), parula(N), turbo(N)

    figure; hold on;

    % --- Decide background ---
    % Show the first image as background
    imshow(images{1}, []);  
    hold on;

    for i = 1:N
        img = images{i};
        if ndims(img) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end

        % --- Extract edges (you can use 'sobel', 'canny', etc.) ---
        e = edge(img_gray, 'sobel');

        % --- Get colored edge matrix ---
        edge_img = zeros([size(img_gray), 3]);   % NxMx3
        for c = 1:3
            edge_img(:,:,c) = e * cmap(i,c);
        end

        % --- Overlay with small transparency ---
        h = imshow(edge_img);
        alpha(h, 0.25);   % tweak transparency if needed
    end

    title('Deformation Progression (Edge Overlays)');
    hold off;
end


function overlay_colorcoded_images(img_files, alpha_values)
    % img_files: cell array of image filenames  { 't1.png', 't2.png', ... }
    % alpha_values: vector of alpha values      [0.3, 0.5, 0.8, ...]

    % If the user only gives 1 alpha, generate automatic alphas
    % if length(alpha_values) == 1
    %     alpha_values = linspace(alpha_values, 1, numel(img_files));
    % end
    % 
    % % If no alphas given, create defaults
    % if nargin < 2
    %     alpha_values = linspace(0.3, 1.0, numel(img_files));
    % end

    % List of colormaps to cycle through
    cmaps = {hot(256), jet(256), parula(256), turbo(256), winter(256), autumn(256)};

    figure; hold on;

    for k = 1:numel(img_files)
        % Read image
        img = img_files{k};

        % Convert to grayscale if not already
        if size(img,3) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end

        % Convert grayscale → indexed → RGB with colormap
        cmap_k = cmaps{ mod(k-1, numel(cmaps)) + 1 };  % cycle colormaps
        img_rgb = ind2rgb(gray2ind(img_gray,256), cmap_k);

        % Display with alpha blending
        h = imshow(img_rgb);
        set(h, 'AlphaData', alpha_values(k));
    end

    hold off;
    title('Color-coded Deformation Overlays (Time Progression)');
end

function overlay_images(images, alphas)
    % images : cell array of image matrices
    % alphas : array of alpha values between [0,1]

    N = numel(images);
    if nargin < 2
        % Default: progressively increase opacity per frame
        alphas = linspace(0.2, 0.6, N);
    end

    im_base = im2double(images{1});
    overlay = im_base;

    for i = 2:N
        img = im2double(images{i});
        a = alphas(i);
        overlay = (1-a)*overlay + a*img;
    end

    figure; imshow(overlay);
    title("Overlay of deformed tyre images");
end

function overlay_color_tinted(images)
    N = numel(images);
    cmap = turbo(N);   % or turbo(N), hot(N)
    alphas = linspace(0.2, 0.6, N);
    size_ref = size(images{1});
    canvas = zeros(size_ref(1), size_ref(2), 3);

    for i = 1:N
        img = im2double(images{i});
        tint = reshape(cmap(i,:), 1, 1, 3);
        a = alphas(i);
        img_tinted = img .* tint;
        img_tinted = (1-a)*img_tinted + a*img;
        canvas = max(canvas, img_tinted);
    end

    figure; imshow(canvas);
    title("Color-tinted deformation overlay");
end

function make_gif(images, outfile)
    for i = 1:numel(images)
        [A,map] = rgb2ind(im2uint8(images{i}),256);
        if i == 1
            imwrite(A,map,outfile,"gif","LoopCount",inf,"DelayTime",0.4);
        else
            imwrite(A,map,outfile,"gif","WriteMode","append","DelayTime",0.4);
        end
    end
end
