// ---------- CONFIGURATION ----------
pixelWidth = 0.1116;        // µm per pixel (500× image)
pixelHeight = 0.1116;
unit = "micron";
gaussianSigma = 1.5;
blocksize = 127;             // CLAHE parameter
histBins = 256;
maxSlope = 3.0;
wekaModelPath = "";          // Set to path if you have a trained model
outDir = getDirectory("Choose Output Directory");

// ---------- STEP 1: Calibration ----------
run("Properties...", "unit="+unit+" pixel_width="+pixelWidth+" pixel_height="+pixelHeight);

// ---------- STEP 2: Preprocessing ----------
run("Gaussian Blur...", "sigma=" + gaussianSigma);
run("Enhance Local Contrast (CLAHE)", 
    "blocksize=" + blocksize + " histogram=" + histBins + " maximum=" + maxSlope);

// ---------- STEP 3: Segmentation ----------
if (wekaModelPath == "") {
    showMessage("Trainable Weka Segmentation",
        "Please train Weka segmentation:\n1. Label Fibers and Background\n2. Train classifier\n3. Click 'Create result' to continue.");
} else {
    run("Trainable Weka Segmentation", "load="+wekaModelPath+" apply");
}

// Pause for manual thresholding
waitForUser("Adjust threshold (Image → Adjust → Threshold) so fibers = green, pores = red, then click OK to continue.");

// ---------- STEP 4: Binarization + Repair ----------
run("Convert to Mask");

// Repair fragmented fibers:
run("Close");                 // connect small gaps
run("Fill Holes");            // fill internal voids
run("Dilate");                // slightly strengthen thin connections
run("Erode");                 // restore single-pixel thickness

// Optional cleanup (remove isolated specks)
run("Remove Outliers...", "radius=2 threshold=50 which=Bright");

// ---------- STEP 5: Skeletonization ----------
run("Skeletonize (2D/3D)");

// ---------- STEP 6: Analyze Skeleton ----------
setOption("BlackBackground", false);
run("Analyze Skeleton (2D/3D)", "prune=none calculate show detailed show_junctions show_endpoints");

// ---------- STEP 7: Export Results ----------
try {
    selectWindow("Results");
    saveAs("Results", outDir + "branches.csv");
} catch (e) { print("⚠️ No 'Results' table found."); }

try {
    selectWindow("Junctions");
    saveAs("Results", outDir + "junctions.csv");
} catch (e) { print("⚠️ No 'Junctions' table found."); }

try {
    selectWindow("Endpoints");
    saveAs("Results", outDir + "endpoints.csv");
} catch (e) { print("⚠️ No 'Endpoints' table found."); }

// ---------- STEP 8: Cleanup ----------
run("Close All");
print("\\n[✓] Completed: CSVs exported to → " + outDir);
showMessage("Done", "Skeleton analysis complete! Files saved to:\n" + outDir);