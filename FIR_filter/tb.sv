module FIR_tb;

    // Parameters
    localparam CLK_PERIOD = 1000; // Clock period in ns
    localparam SIM_TIME = 20000000; // Simulation time in ns
    localparam AMPLITUDE = 32767; // Amplitude of the sine wave
    
    // Inputs
    logic clk;
    logic reset;
    logic [32:0] x;
    
    // Outputs
    logic  signed[32:0] y;
    logic [32:0] input_data[99999:0] ;
    integer data_index = 0;
    // Instantiate the FIR filter module
    FIR dut (
        .clk(clk),
        .reset(reset),
        .x(x),
        .y(y)
    );
    
    // Clock generation
    always #((CLK_PERIOD / 2)) clk = ~clk;
    
    // Reset generation
    initial begin
        reset = 1;
        clk = 0;
        #((10 * CLK_PERIOD)); // Hold reset for 10 clock cycles
        reset = 0;
        
    end
    
    initial begin
        $readmemh("sin.txt", input_data);
    end
    // Input signal generation (sine wave)
    initial begin
        //x = 0;

        forever begin
            // Generate sine wave with a frequency of 10000 Hz
            if (data_index < 99999) begin

                x = input_data[data_index]-32'd65536;
                $display("Input: %d", input_data[data_index]);
                data_index = data_index + 1;
                #(1*CLK_PERIOD);
            end
            else $finish;
        end
    end
    
    // Display output
    always @(posedge clk) begin
        $display("Output: %d", y);
        
    end
    
endmodule
