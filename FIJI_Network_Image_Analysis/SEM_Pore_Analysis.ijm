// ====================================================================
// SEM Pore Size Analysis - CSV only (universal IJM)
// - Bright scaffold, dark pores (SEM)
// - Uses SEM calibration as-is
// - Outputs one CSV per image with:
//     Area (um^2), EqDiam_um, plus shape metrics
// ====================================================================

macro "SEM_Pore_Area_And_Diameter_CSV_Only" {

    // ----- Choose folders -----
    inputDir = getDirectory("Choose folder with SEM images");

    outputDir = getDirectory("Choose output folder for CSVs");
    
    list = getFileList(inputDir);

    for (i = 0; i < list.length; i++) {

        name = list[i];

        // Only process image files
        if (!(endsWith(name, ".tif") || endsWith(name, ".tiff") ||
              endsWith(name, ".png") || endsWith(name, ".jpg") ||
              endsWith(name, ".jpeg")))
            continue;

        open(inputDir + name);
        title = getTitle();
        print("\\nProcessing: " + title);

        // ----------------------------
        // Preprocessing
        // ----------------------------
        run("8-bit");
        run("Median...", "radius=2");
        run("Remove Outliers...", "radius=2 threshold=50 which=Dark");
        run("Gaussian Blur...", "sigma=1");

        // Threshold pores (dark)
        setAutoThreshold("Default dark");
        setOption("BlackBackground", false);
        run("Convert to Mask");

        // Pores as white particles
        run("Invert");
        run("Open");
        run("Close");

        // ----------------------------
        // Analyze particles
        // ----------------------------
        minArea = 1.0;  // minimum pore area in um^2

        run("Set Measurements...",
            "area perimeter shape feret redirect=None decimal=4");

        if (isOpen("Results")) {
            selectWindow("Results");
            run("Clear Results");
        }

        run("Analyze Particles...",
            "size=" + minArea + "-Infinity circularity=0-1 show=Nothing display clear exclude");

        n = nResults;
        if (n == 0) {
            print("No pores detected above " + minArea + " um^2.");
            close();
            continue;
        }

        // ----------------------------
        // Compute equivalent diameter
        // ----------------------------
        for (r = 0; r < n; r++) {
            A = getResult("Area", r);       // um^2
            eqD = 2 * sqrt(A / PI);         // um
            setResult("EqDiam_um", r, eqD);
            setResult("Image", r, title);
        }
        updateResults();

        // ----------------------------
        // Build base name manually
        // ----------------------------
        dot = lastIndexOf(title, ".");
        if (dot < 0)
            base = title;
        else
            base = substring(title, 0, dot);

        // ----------------------------
        // Save CSV
        // ----------------------------
        csvPath = outputDir + base + "_pore_sizes.csv";
        print("Saving CSV -> " + csvPath);
        saveAs("Results", csvPath);

        // Cleanup for next image
        selectWindow("Results");
        run("Clear Results");
        close(); // close Results window
        close(); // close image
    }

    print("\\nDONE. All pore CSVs saved to:");
    print(outputDir);
}
