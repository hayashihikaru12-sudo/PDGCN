from dataclasses import dataclass


@dataclass(frozen=True)
class PDGCNConfig:
    node_input_size: int = 8
    edge_input_size: int = 7
    global_input_size: int = 1
    hidden_size: int = 64
    message_passing_num: int = 3
    output_size: int = 1

    gamma_upwind: float = 0.8
    use_aniso_gate: bool = True
    include_global: bool = True

    dropout: float = 0.0
    layer_norm: bool = True

    lambda_outflow: float = 1.0
    inverse_pe: float = 1.0
    pi_q: float = 1.0
    k_ratio: float = 0.05
    dt_star: float = 1.0
    gradient_regularization: float = 0.0
    dirichlet_temperature_star: float = 0.0

    @property
    def encoder_node_input_size(self) -> int:
        if self.include_global:
            return self.node_input_size + self.global_input_size
        return self.node_input_size

    def __post_init__(self):
        positive_ints = (
            "node_input_size",
            "edge_input_size",
            "global_input_size",
            "hidden_size",
            "message_passing_num",
            "output_size",
        )
        for field_name in positive_ints:
            value = int(getattr(self, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")
        if float(self.lambda_outflow) < 0:
            raise ValueError(f"lambda_outflow must be non-negative, got {self.lambda_outflow}.")
        if float(self.inverse_pe) < 0:
            raise ValueError(f"inverse_pe must be non-negative, got {self.inverse_pe}.")
        if float(self.pi_q) < 0:
            raise ValueError(f"pi_q must be non-negative, got {self.pi_q}.")
        if float(self.k_ratio) < 0:
            raise ValueError(f"k_ratio must be non-negative, got {self.k_ratio}.")
        if float(self.dt_star) <= 0:
            raise ValueError(f"dt_star must be positive, got {self.dt_star}.")
        if float(self.gradient_regularization) < 0:
            raise ValueError(
                f"gradient_regularization must be non-negative, got {self.gradient_regularization}."
            )
