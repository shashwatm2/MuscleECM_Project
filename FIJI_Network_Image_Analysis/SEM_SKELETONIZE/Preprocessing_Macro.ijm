// ---------- USER PARAMETERS ----------

// Pixel calibration
pixel_size_um = 0.01116;   // microns per pixel (EDIT THIS)
unit = "micron";

// CLAHE parameters (SEM-friendly)
clahe_blocksize = 127;
clahe_histBins  = 256;
clahe_maxSlope  = 3.0;

// Gaussian blur (pixels)
gauss_sigma = 1.5;

// Minimum object size to keep (pixels)
min_particle_size = 50;

// Fibers brighter than background?
fibers_are_bright = true;

// ---------- START ----------

setBatchMode(true);

// Ensure grayscale
run("8-bit");

// Set spatial calibration
run("Set Scale...", 
    "distance=1 known=" + pixel_size_um + " unit=" + unit + " global");

// Contrast enhancement (CLAHE)
run("Enhance Local Contrast (CLAHE)", 
    "blocksize=" + clahe_blocksize +
    " histogram=" + clahe_histBins +
    " maximum=" + clahe_maxSlope);

// Light denoising
run("Gaussian Blur...", "sigma=" + gauss_sigma);

// Clear a 3-pixel border to prevent boundary skeletons
border = 3;
makeRectangle(border, border,
              getWidth() - 2*border,
              getHeight() - 2*border);
run("Clear Outside");
