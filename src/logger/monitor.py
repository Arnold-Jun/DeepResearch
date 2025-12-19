from rich.text import Text
from pydantic import BaseModel, computed_field


class TokenUsage(BaseModel):
    """
    包含给定步骤或运行的 token 使用信息。
    """

    input_tokens: int
    output_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Timing(BaseModel):
    """
    包含给定步骤或运行的时序信息。
    """

    start_time: float
    end_time: float | None = None

    @computed_field
    @property
    def duration(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.start_time

    def __repr__(self) -> str:
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"

class Monitor:
    def __init__(self, tracked_model, logger):
        self.step_durations = []
        self.tracked_model = tracked_model
        self.logger = logger
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def get_total_token_counts(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.total_input_token_count,
            output_tokens=self.total_output_token_count,
        )

    def reset(self):
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """更新监控器的指标。

        参数:
            step_log ([`MemoryStep`]): 用于更新监控器的步骤日志。
        """
        step_duration = step_log.timing.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        if step_log.token_usage is not None:
            self.total_input_token_count += step_log.token_usage.input_tokens
            self.total_output_token_count += step_log.token_usage.output_tokens
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)