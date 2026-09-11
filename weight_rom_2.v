`timescale 1ns / 1ps

module weight_rom_2 #(
    parameter NUM_FILTERS = 8,
    parameter DATA_WIDTH = 8,
    parameter WEIGHTS_FILE = "../export/weights.mem"
)(
    input wire clk, 
    input wire [((NUM_FILTERS <= 1) ? 1 : $clog2(NUM_FILTERS))-1:0] addr,
    output reg [(9*DATA_WIDTH)-1:0] data_out 
);

    // 72-bit wide ROM.  "ramstyle" is the Quartus attribute; the original
    // Xilinx-specific "rom_style" attribute is ignored by Quartus.
    (* ramstyle = "M9K" *)
    reg [(9*DATA_WIDTH)-1:0] rom [0:NUM_FILTERS-1];

    // The Quartus project file lives in ../par, so this resolves to the
    // repository's export directory rather than a directory under par.
    initial begin
        $readmemh(WEIGHTS_FILE, rom);
    end

    // SYNCHRONOUS READ (Clocked)
    always @(posedge clk) begin
        data_out <= rom[addr];
    end

endmodule
