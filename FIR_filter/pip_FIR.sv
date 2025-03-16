module pip_FIR (
    input logic clk,
    input logic reset,
    input logic signed [32:0] x,
    output logic signed [32:0] y
);

// Coefficients for the FIR filter
parameter [32:0] coeff[128:0] = 
    '{4,13,30,57,91,126,146,134,70,-62,-266,-527,-811,-1069,-1244,-1288,-1174,-908,-536,-131,219,438,482,353,104,-179,-400,
    -487,-409,-195,81,320,438,390,192,-83,-333,-460,-410,-195,108,384,520,452,193,-168,-493,-642,-537,-192,281,701,883,711,192,
    -523,-1176,-1472,-1174,-191,1376,3245,5017,6284,6744,6284,5017,3245,1376,-191,-1174,-1472,-1176,-523,192,711,883,701,281,
    -192,-537,-642,-493,-168,193,452,520,384,108,-195,-410,-460,-333,-83,192,390,438,320,81,-195,-409,-487,-400,-179,104,353,
    482,438,219,-131,-536,-908,-1174,-1288,-1244,-1069,-811,-527,-266,-62,70,134,146,126,91,57,30,13,4};
// Internal pipeline registers
logic signed [32:0] delay [127:0];
logic signed [32:0] line_out [128:0];
logic signed [32:0] sum [127:0];

integer i,j,k;

always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
        delay <= '{default: 32'd0};
        line_out <= '{default: 32'd0};
        sum <= '{default: 32'd0};
    end else begin
        delay[0] <= x;
        sum[0] = line_out[0] + line_out[1];
        line_out[0] <= x*delay[0];
        for (i=1;i<128;i=i+1)begin
            delay[i] <= delay[i-1] ;
            sum[i] = sum[i-1] + line_out[i+1];
        end
        line_out[128] <= x * coeff[0];
        for (k=0;k<128;k=k+1)begin
            line_out[k] <= delay[127-k] * coeff[128-k];
        end  
        end
 
    end



assign y = sum[127];

endmodule
