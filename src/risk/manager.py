from __future__ import annotations

from datetime import datetime, timezone

from src.config_runtime.models import ExecutionConfig, OpportunityConfig, RiskConfig
from src.domain.models import AccountSnapshot, OpportunityCandidate, RejectionReason, RiskDecision, RiskStatus
from src.risk import rules


class RiskManager:
    def __init__(self, risk_config: RiskConfig, opportunity_config: OpportunityConfig, execution_config: ExecutionConfig):
        self.risk_config = risk_config
        self.opportunity_config = opportunity_config
        self.execution_config = execution_config

    def evaluate(self, candidate: OpportunityCandidate, account: AccountSnapshot) -> RiskDecision:
        if not self.risk_config.enabled:
            return RiskDecision(
                candidate_id=candidate.candidate_id,
                status=RiskStatus.APPROVED,
                approved_notional_usd=candidate.target_notional_usd,
                ts=datetime.now(timezone.utc),
            )

        checks = [
            rules.check_score(candidate, self.risk_config),
            rules.check_net_profit(candidate, self.opportunity_config),
            rules.check_depth(candidate, self.risk_config),
            rules.check_book_age(candidate, self.opportunity_config),
            rules.check_market_whitelist(candidate, self.risk_config),
            rules.check_order_size(candidate, self.risk_config, self.execution_config),
            rules.check_daily_loss(account, self.risk_config),
            rules.check_consecutive_losses(account, self.risk_config),
            rules.check_open_positions(account, self.risk_config),
            rules.check_market_exposure(candidate, account, self.risk_config),
            rules.check_total_exposure(candidate, account, self.risk_config),
        ]

        failures = [check for check in checks if not check.passed]
        reason_codes = [check.code for check in failures]
        metadata = {
            "checks": [
                {
                    "code": check.code,
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in checks
            ]
        }
        halt = any(
            code in {
                RejectionReason.DAILY_LOSS_LIMIT.value,
                RejectionReason.CONSECUTIVE_LOSS_LIMIT.value,
            }
            for code in reason_codes
        )

        return RiskDecision(
            candidate_id=candidate.candidate_id,
            status=RiskStatus.HALTED if halt else (RiskStatus.BLOCKED if failures else RiskStatus.APPROVED),
            approved_notional_usd=0.0 if failures else candidate.target_notional_usd,
            reason_codes=reason_codes,
            human_review_required=self.execution_config.mode == "live" and self.risk_config.require_human_confirmation_for_live,
            halt_trading=halt,
            metadata=metadata,
            ts=datetime.now(timezone.utc),
        )
