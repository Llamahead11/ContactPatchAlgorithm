function [boundaries, shapes, labels] = find_lug_boundaries(input2D, varargin)
% FIND_LUG_BOUNDARIES  Find lug boundaries in a 2D contact patch.
%
% USAGE
%   [boundaries, shapes, labels] = find_lug_boundaries(input2D)
%   [boundaries, shapes, labels] = find_lug_boundaries(..., 'eps',0.002, 'MinPts',8, 'Alpha',[], 'Plot',true)
%
% INPUT
%   input2D : either
%       - Nx2 numeric matrix of [x y] points (units: meters or pixels), or
%       - 2D binary image (logical or numeric) where true/1 are contact patch pixels.
%
% OPTIONAL NAME-VALUE PAIRS
%   'eps'     - DBSCAN epsilon. If empty, estimated from median nearest-neighbour distance.
%   'MinPts'  - DBSCAN MinPts. Default 8.
%   'Alpha'   - scalar alpha value to use for alphaShape (if empty each cluster estimates its own).
%   'Plot'    - true/false (default true) to show results.
%   'Shrink'  - fallback shrink factor for boundary() if alphaShape fails (default 0.8).
%
% OUTPUT
%   boundaries : cell array; each cell contains an Mx2 polygon (x,y) boundary of a lug.
%                A cluster may produce multiple polygons (multiple cells) if disconnected regions exist.
%   shapes     : cell array of alphaShape objects (or empty for clusters that fell back to boundary).
%   labels     : cluster labels for each input point (Nx1, with -1 = noise).
%
% REQUIREMENTS
%   Statistics and Machine Learning Toolbox for dbscan / knnsearch.
%   (alphaShape is built-in in base MATLAB.)
%
% EXAMPLE
%   % For point cloud:
%   pts = contactPoints; % Nx2
%   [B,S,labels] = find_lug_boundaries(pts, 'Plot', true);
%
%   % For binary image:
%   BW = imread('contact_mask.png') > 0;
%   [B,S,labels] = find_lug_boundaries(BW, 'Plot', true);

% Parse inputs
p = inputParser;
p.addRequired('input2D');
p.addParameter('eps', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x>0));
p.addParameter('MinPts', 8, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('Alpha', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x>0));
p.addParameter('Plot', true, @(x) islogical(x));
p.addParameter('Shrink', 0.8, @(x) isnumeric(x) && x>0 && x<=1);
p.parse(input2D, varargin{:});
opts = p.Results;

%% Convert input to Nx2 point list (x,y)
if ismatrix(input2D) && size(input2D,2) == 2 && isnumeric(input2D)
    pts = double(input2D);
elseif ismatrix(input2D) && (islogical(input2D) || isnumeric(input2D))
    % assume binary image / mask
    BW = logical(input2D);
    [r,c] = find(BW);
    % Convert to x,y coordinates: depending on your coordinate convention,
    % you may want (x=c, y=r) or real-world scaling. We'll use x=c, y=r.
    pts = [double(c), double(r)];
else
    error('Unsupported input type. Provide Nx2 numeric points or a 2D binary image.');
end

N = size(pts,1);
if N < 3
    error('Not enough points to form boundaries.');
end

%% Estimate default eps if not set
if isempty(opts.eps)
    % Use knnsearch to estimate typical nearest-neighbour distance.
    k = min(4, N); % use up to 4 neighbors (including self)
    [~, D] = knnsearch(pts, pts, 'K', k);
    % D(:,1) is zero (self); use second column as nearest real neighbor when possible
    if k >= 2
        nnDist = D(:,2);
    else
        nnDist = D(:,1);
    end
    medNN = median(nnDist);
    % eps: a few times the median nearest-neighbor to connect points of a lug
    opts.eps = max( eps(1), 3 * medNN ); %#ok<NASGU> % ensure numeric
    % If medNN is 0 (e.g., integer pixel coords with duplicates), use small default
    if medNN == 0
        opts.eps = 1.5;
    end
end

%% Run DBSCAN clustering
labels = dbscan(pts, opts.eps, opts.MinPts);

uniqueLabels = setdiff(unique(labels), -1); % ignore noise label -1
nClusters = numel(uniqueLabels);

boundaries = {};
shapes = cell(nClusters,1);

% colors for plotting if requested
if opts.Plot
    figure; hold on; axis equal; view(2);
    title('Detected lug boundaries');
    xlabel('X'); ylabel('Y');
    cmap = lines(max(1,nClusters));
end

clusterCount = 0;
for i = 1:nClusters
    lab = uniqueLabels(i);
    clIdx = (labels == lab);
    clusterPts = pts(clIdx, :);

    % If cluster too small, skip
    if size(clusterPts,1) < 3
        continue;
    end

    % Decide alpha: use given Alpha or estimate per-cluster
    if ~isempty(opts.Alpha)
        alphaVal = opts.Alpha;
    else
        % estimate using median nearest-neighbour inside cluster
        kc = min(4,size(clusterPts,1));
        [~, Dc] = knnsearch(clusterPts, clusterPts, 'K', kc);
        if kc >= 2
            medc = median(Dc(:,2));
        else
            medc = median(Dc(:,1));
        end
        % safety if medc==0 (duplicated points)
        if medc == 0
            medc = mean(std(clusterPts),2) / 10 + eps;
        end
        alphaVal = 1.5 * medc; % factor can be tuned
    end

    % Build alpha shape
    shp = alphaShape(clusterPts(:,1), clusterPts(:,2), alphaVal);

    % If alpha is too small the shape may have zero area or many regions.
    % Try to increase alpha until shape has area > 0 and at least 1 boundary.
    attempt = 0;
    while (isempty(shp) || shp.area == 0) && attempt < 6
        alphaVal = alphaVal * 2;
        shp = alphaShape(clusterPts(:,1), clusterPts(:,2), alphaVal);
        attempt = attempt + 1;
    end

    shapes{i} = shp;

    % Extract boundary facets (edges). For 2D alphaShape, boundaryFacets returns edges
    try
        facets = boundaryFacets(shp); % Mx2 indices into clusterPts (1-based)
    catch
        facets = [];
    end

    polys = {}; % may be multiple loops

    if ~isempty(facets)
        % Convert facets to ordered polygons (there may be multiple disjoint loops)
        polys = edges_to_polygons(facets, clusterPts);
    end

    % If alpha failed (no facets or degenerate), fallback to boundary()
    if isempty(polys)
        try
            kShrink = opts.Shrink;
            bIdx = boundary(clusterPts(:,1), clusterPts(:,2), kShrink);
            polys = { clusterPts(bIdx, :) };
            shapes{i} = []; % mark shape empty to indicate fallback
        catch
            % ultimate fallback: convex hull
            k = convhull(clusterPts(:,1), clusterPts(:,2));
            polys = { clusterPts(k, :) };
            shapes{i} = []; 
        end
    end

    % Append polygons (sometimes a cluster yields multiple polygons)
    for pIdx = 1:numel(polys)
        clusterCount = clusterCount + 1;
        boundaries{clusterCount} = polys{pIdx};
        if opts.Plot
            % plot polygon boundary
            c = cmap(mod(clusterCount-1,size(cmap,1))+1,:);
            plot(polys{pIdx}(:,1), polys{pIdx}(:,2), '-', 'LineWidth', 1.6, 'Color', c);
            % optionally plot cluster points lightly
            hold on
            scatter(clusterPts(:,1), clusterPts(:,2), 6, c, 'filled', 'MarkerFaceAlpha',0.12);
            hold on
        end
    end
end

if opts.Plot
    axis tight;
    grid on
    hold off;
end

end

%% Helper: convert edge list to ordered polygon cycles
function polygons = edges_to_polygons(facets, pts)
% facets: Mx2 list of vertex indices into pts
% pts: Kx2 coordinates
polygons = {};
if isempty(facets), return; end

% Build adjacency map: for each vertex, list connected vertices and edge indices
edges = facets;
M = size(edges,1);
used = false(M,1);

% adjacency: containers.Map would be slower; use cell array indexed by vertex
nPts = max(edges(:));
adj = cell(nPts,1);
for e = 1:M
    v1 = edges(e,1);
    v2 = edges(e,2);

    if isempty(adj{v1})
        adj{v1} = { struct('nbr',v2,'eid',e) };
    else
        adj{v1}{end+1} = struct('nbr',v2,'eid',e);
    end

    if isempty(adj{v2})
        adj{v2} = { struct('nbr',v1,'eid',e) };
    else
        adj{v2}{end+1} = struct('nbr',v1,'eid',e);
    end
end

for startEdge = 1:M
    if used(startEdge), continue; end

    % pick an unused edge, start building a cycle
    e = startEdge;
    used(e) = true;
    v_start = edges(e,1);
    v_next = edges(e,2);
    polyIdx = [v_start, v_next];

    curr = v_next;
    prev = v_start;

    % follow edges until we return to start or stuck
    loopGuard = 0;
    while curr ~= v_start && loopGuard < 3*M
        neighbors = adj{curr};
        % choose next neighbor that is not prev and whose edge is unused if possible
        chosen = [];
        for k = 1:numel(neighbors)
            nb = neighbors{k};
            if nb.nbr == prev
                continue;
            end
        end
        if isempty(chosen)
            % try any neighbor (maybe we must re-use an edge)
            for k = 1:numel(neighbors)
                nb = neighbors{k};
                if nb.nbr ~= prev
                    chosen = nb;
                    break;
                end
            end
        end
        if isempty(chosen)
            break; % dead end
        end
        % mark edge used if possible
        used(chosen.eid) = true;
        prev = curr;
        curr = chosen.nbr;
        polyIdx(end+1) = curr;
        loopGuard = loopGuard + 1;
    end

    % If we ended with a closed loop (first == last) accept polygon
    if numel(polyIdx) >= 3 && polyIdx(end) == polyIdx(1)
        coords = pts(polyIdx, :);
        polygons{end+1} = coords;
    else
        % If not closed, attempt to close by trimming duplicates and appending start
        if numel(polyIdx) >= 3
            if polyIdx(1) ~= polyIdx(end)
                polyIdx(end+1) = polyIdx(1);
            end
            polygons{end+1} = pts(polyIdx,:);
        end
    end
end

% Finally, remove duplicates (same polygon reversed etc.)
% (simple uniqueness check by first point sequence)
uniq = {};
final = {};
for i=1:numel(polygons)
    P = polygons{i};
    key1 = mat2str(round(P(:,1:2),6)); % coarse string key
    if ~any(strcmp(key1, uniq))
        uniq{end+1} = key1;
        final{end+1} = P;
    end
end
polygons = final;

end
