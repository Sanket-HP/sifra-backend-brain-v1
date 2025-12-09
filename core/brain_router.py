# ============================================================
#  SIFRA BRAIN ROUTER v4.5 — Unified Brain Router 3.0
#
#  New Capabilities:
#    ✔ CRE reasoning summary
#    ✔ DMAO agent routing summary
#    ✔ ALL learning state returned
#    ✔ Automatic formatting for dashboards & APIs
#    ✔ Safer NARE-X natural language extraction
#    ✔ Error-proof unified output wrapper
# ============================================================

from core.sifra_core import SifraCore


class SifraBrain:
    """
    Unified SIFRA Brain Interface (v4.5)
    ------------------------------------------------------------
    Responsibilities:
      • Accept any goal + dataset
      • Execute SifraCore (HDP + HDS + CRE + DMAO + ALL)
      • Return:
          - Raw brain output
          - Human-readable NARE-X explanation
          - CRE reasoning summary
          - DMAO agent path
          - Learning updates
    """

    def __init__(self):
        self.brain = SifraCore()
        print("[SIFRA BRAIN] Unified Brain Router v4.5 ready.")

    # --------------------------------------------------------
    #  MAIN ENTRY
    # --------------------------------------------------------
    def run(self, goal, dataset):
        """
        Entry point for:
            - Web dashboards
            - Mobile apps
            - APIs
            - External clients
            - Third-party integrations
        """

        core_output = self.brain.run(goal, dataset)

        # -----------------------------------------------
        # SAFE EXTRACTORS
        # -----------------------------------------------
        narex_block = core_output.get("DMAO", {}).get("agent_output", {})
        cre_block = core_output.get("CRE", {})
        dmao_block = core_output.get("DMAO", {})
        learn_block = core_output.get("ALL", {})

        # Natural language output from NARE-X (safe fallback)
        natural = ""
        if isinstance(narex_block, dict):
            natural = narex_block.get("natural_language_response", "") or ""
        else:
            natural = ""

        # CRE reasoning summary (compressed)
        cre_summary = cre_block.get("final_decision", "")

        # DMAO agent path
        agent_used = dmao_block.get("agent_selected", "Unknown")

        # -------------------------------------------------
        # UNIFIED OUTPUT WRAPPER
        # -------------------------------------------------
        return {
            "status": "success",
            "goal": goal,
            "brain_mode": core_output.get("brain_mode", ""),

            # -----------------------------
            # RAW OUTPUT
            # -----------------------------
            "raw_output": core_output,

            # -----------------------------
            # HUMAN UNDERSTANDABLE OUTPUT
            # -----------------------------
            "natural_language_response": natural,
            "reasoning_summary": cre_summary,
            "agent_used": agent_used,

            # -----------------------------
            # ADAPTIVE LEARNING FEEDBACK
            # -----------------------------
            "learning_update": learn_block,

            # -----------------------------
            # UNIVERSAL MESSAGE
            # -----------------------------
            "message": "SIFRA Unified Brain (v4.5) processed the request successfully."
        }
# ============================================================
#  SIFRA ENGINE ROUTER v4.5 — CRE + DMAO + ALL