`timescale 1ns / 1ps

// Resource-efficient fully connected classifier. Pooling values are captured
// first; a single signed MAC then evaluates one class at a time.
// FC_WEIGHTS_FILE stores NUM_CLASSES consecutive FEATURE_COUNT-value rows.
module fc_classifier #(
    parameter FEATURE_COUNT   = 1352,
    parameter NUM_CLASSES     = 10,
    parameter FEATURE_WIDTH   = 32,
    parameter WEIGHT_WIDTH    = 8,
    parameter ACC_WIDTH       = 48,
    parameter FC_WEIGHTS_FILE = "../export/fc_weights.mem"
)(
    input  wire                            clk,
    input  wire                            rst_n,
    input  wire                            feature_valid,
    input  wire signed [FEATURE_WIDTH-1:0] feature_data,
    input  wire                            start,
    output reg                             class_valid,
    output reg [((NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES))-1:0]
                                         predicted_class,
    output reg signed [ACC_WIDTH-1:0]     class_score,
    output reg                             done
);
    localparam CLASS_WIDTH = (NUM_CLASSES <= 1) ? 1 : $clog2(NUM_CLASSES);
    localparam FEATURE_INDEX_WIDTH = (FEATURE_COUNT <= 1) ? 1 : $clog2(FEATURE_COUNT);
    localparam CAPTURE = 1'b0;
    localparam CALCULATE = 1'b1;

    reg state;
    reg signed [FEATURE_WIDTH-1:0] feature_mem [0:FEATURE_COUNT-1];
    reg signed [WEIGHT_WIDTH-1:0] weights [0:NUM_CLASSES*FEATURE_COUNT-1];
    reg [FEATURE_INDEX_WIDTH-1:0] capture_index;
    reg [FEATURE_INDEX_WIDTH-1:0] feature_index;
    reg [CLASS_WIDTH-1:0] class_index;
    reg signed [ACC_WIDTH-1:0] accumulator;
    reg signed [ACC_WIDTH-1:0] best_score;
    reg [CLASS_WIDTH-1:0] best_class;
    reg signed [ACC_WIDTH-1:0] candidate_score;

    initial begin
        $readmemh(FC_WEIGHTS_FILE, weights);
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state           <= CAPTURE;
            capture_index   <= 0;
            feature_index   <= 0;
            class_index     <= 0;
            accumulator     <= 0;
            best_score      <= 0;
            best_class      <= 0;
            class_valid     <= 0;
            predicted_class <= 0;
            class_score     <= 0;
            done            <= 0;
        end else begin
            class_valid <= 0;
            done        <= 0;

            if (state == CAPTURE) begin
                if (feature_valid) begin
                    feature_mem[capture_index] <= feature_data;
                    if (capture_index < FEATURE_COUNT - 1)
                        capture_index <= capture_index + 1'b1;
                end

                // pool_done is asserted with the final feature_valid pulse.
                // Calculation begins next cycle, after that feature write.
                if (start) begin
                    state         <= CALCULATE;
                    feature_index <= 0;
                    class_index   <= 0;
                    accumulator   <= 0;
                    best_score    <= 0;
                    best_class    <= 0;
                end
            end else begin
                candidate_score = accumulator
                    + feature_mem[feature_index]
                    * weights[class_index*FEATURE_COUNT + feature_index];

                if (feature_index == FEATURE_COUNT - 1) begin
                    if (class_index == 0 || candidate_score > best_score) begin
                        best_score <= candidate_score;
                        best_class <= class_index;
                    end

                    if (class_index == NUM_CLASSES - 1) begin
                        if (class_index == 0 || candidate_score > best_score) begin
                            class_score     <= candidate_score;
                            predicted_class <= class_index;
                        end else begin
                            class_score     <= best_score;
                            predicted_class <= best_class;
                        end
                        class_valid   <= 1;
                        done          <= 1;
                        state         <= CAPTURE;
                        capture_index <= 0;
                        feature_index <= 0;
                        class_index   <= 0;
                        accumulator   <= 0;
                    end else begin
                        class_index   <= class_index + 1'b1;
                        feature_index <= 0;
                        accumulator   <= 0;
                    end
                end else begin
                    accumulator   <= candidate_score;
                    feature_index <= feature_index + 1'b1;
                end
            end
        end
    end
endmodule
