import processing.serial.*;

Serial myPort;
int numPoints = 1024; // Must be a power of 2 for the manual FFT

// Arrays for the three stages of our data
float[] rawData = new float[numPoints];
float[] filteredData = new float[numPoints];
float[] spectrum = new float[numPoints / 2]; 

float[][] Fdata = new float[numPoints][2];
float sampleRate = 250.0; 

void setup() {
  size(1024, 900); // Taller window to fit all 3 plots comfortably
  // Make sure this matches your actual COM port!
  myPort = new Serial(this, "COM10", 115200); 
  myPort.bufferUntil('\n');
}

void draw() {
  background(20);

  // --- 1. PROCESS THE DATA (FFT -> FILTER -> INVERSE FFT) ---
  processFrequencyDomain();

  // --- 2. DRAW PLOT 1: RAW TIME DOMAIN (Top) ---
  stroke(150); // Gray for raw data
  strokeWeight(2);
  noFill();
  beginShape();
  for (int i = 0; i < numPoints; i++) {
    float x = map(i, 0, numPoints, 50, width-50);
    // Map 0-4095 to the top third of the screen (y: 250 to 50)
    float y = map(rawData[i], 0, 4095, 250, 50); 
    vertex(x, y);
  }
  endShape();

  // --- 3. DRAW PLOT 2: FILTERED TIME DOMAIN (Middle) ---
  stroke(0, 255, 100); // Bright green for filtered data
  beginShape();
  for (int i = 0; i < numPoints; i++) {
    float x = map(i, 0, numPoints, 50, width-50);
    // Map 0-4095 to the middle third of the screen (y: 550 to 350)
    float y = map(filteredData[i], 0, 4095, 550, 350); 
    vertex(x, y);
  }
  endShape();

  // --- 4. DRAW PLOT 3: FREQUENCY SPECTRUM (Bottom) ---
  stroke(255, 100, 100); // Red for frequency spectrum
  strokeWeight(1);
  for (int i = 0; i < numPoints / 4; i++) { // Show 0-62.5Hz
    float x = map(i, 0, numPoints / 4, 50, width-50);
    float h = spectrum[i] * 5; // Adjust multiplier to make spikes visible
    line(x, 850, x, 850 - h);
  }

  // --- UI Labels & Gridlines ---
  fill(255);
  textSize(16);
  
  text("1. RAW SIGNAL (Direct from Arduino)", 50, 40);
  stroke(50); line(0, 280, width, 280); // Divider
  
  text("2. FILTERED SIGNAL (Inverse FFT output, < 40Hz)", 50, 330);
  stroke(50); line(0, 580, width, 580); // Divider
  
  text("3. FREQUENCY SPECTRUM (After Filter is applied)", 50, 620);
  textSize(12);
  text("0Hz", 50, 870);
  text("62.5Hz", width-100, 870);
}

void serialEvent(Serial port) {
  String inData = port.readStringUntil('\n');
  if (inData != null) {
    float val = float(trim(inData));
    if (!Float.isNaN(val)) {
      // Shift raw array and add new point
      for (int i = 0; i < numPoints - 1; i++) {
        rawData[i] = rawData[i+1];
      }
      rawData[numPoints-1] = val;
    }
  }
}

// -----------------------------------------------------------------
// DIGITAL SIGNAL PROCESSING (Frequency Domain Filtering)
// -----------------------------------------------------------------

void processFrequencyDomain() {
  // 1. Load Raw Data into Complex Array
  for (int i = 0; i < numPoints; i++) {
    Fdata[i][0] = rawData[i]; // Real part
    Fdata[i][1] = 0;          // Imaginary part
  }
  
  // 2. Forward FFT (Time -> Frequency)
  performFFT(1, 10); 
  
  float freqRes = sampleRate / numPoints; // ~0.244 Hz per bin
  
  // 3. Analyze and Apply Low-Pass Filter
  for (int i = 0; i < numPoints; i++) {
    // Calculate the actual frequency of this bin
    float freq = i * freqRes;
    
    // Account for FFT symmetry (frequencies wrap around after Nyquist)
    if (freq > sampleRate / 2) {
      freq = sampleRate - freq; 
    }
    
    // Save magnitude to spectrum array for drawing (only need first half)
    if (i < numPoints / 2) {
      // Notice we extract the spectrum *before* we zero it out so you can see 
      // where the filter is applying, or *after* depending on what you want.
      // Here, we'll map what is actually left over.
    }

    // LOW PASS FILTER: If frequency is above 40Hz, destroy it (set to 0)
    if (freq > 40.0) {
      Fdata[i][0] = 0;
      Fdata[i][1] = 0;
    }
    
    // Update spectrum array *after* filtering so the graph shows the cut-off
    if (i < numPoints / 2) {
      spectrum[i] = sqrt(pow(Fdata[i][0], 2) + pow(Fdata[i][1], 2));
    }
  }
  
  // 4. Inverse FFT (Frequency -> Filtered Time)
  performFFT(-1, 10);
  
  // 5. Extract the newly filtered real data
  for (int i = 0; i < numPoints; i++) {
    filteredData[i] = Fdata[i][0];
  }
}

// The manual FFT algorithm
void performFFT(int dir, int m) {
  int n, i, i1, j, k, i2, l, l1, l2;
  float c1, c2, tx, ty, t1, t2, u1, u2, z;
  n = 1;
  for (i=0; i<m; i++) n *= 2;
  i2 = n >> 1;
  j = 0;
  for (i=0; i<n-1; i++) {
    if (i < j) {
      tx = Fdata[i][0]; ty = Fdata[i][1];
      Fdata[i][0] = Fdata[j][0]; Fdata[i][1] = Fdata[j][1];
      Fdata[j][0] = tx; Fdata[j][1] = ty;
    }
    k = i2;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  c1 = -1.0; c2 = 0.0; l2 = 1;
  for (l=0; l<m; l++) {
    l1 = l2; l2 <<= 1; u1 = 1.0; u2 = 0.0;
    for (j=0; j<l1; j++) {
      for (i=j; i<n; i+=l2) {
        i1 = i + l1;
        t1 = u1 * Fdata[i1][0] - u2 * Fdata[i1][1];
        t2 = u1 * Fdata[i1][1] + u2 * Fdata[i1][0];
        Fdata[i1][0] = Fdata[i][0] - t1; Fdata[i1][1] = Fdata[i][1] - t2;
        Fdata[i][0] += t1; Fdata[i][1] += t2;
      }
      z = u1 * c1 - u2 * c2; u2 = u1 * c2 + u2 * c1; u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1) c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }
  if (dir == 1) {
    for (i = 0; i < n; i++) {
      Fdata[i][0] /= n;
      Fdata[i][1] /= n;
    }
  }
}
