using System;

namespace MBBO.Savant
{
    public class GameVector
    {
        // Basic Game Info
        public string Game_Date { get; set; }
        public int Game_PK { get; set; }
        public string Home_Team_Abbr { get; set; }
        public string Away_Team_Abbr { get; set; }
        public int Park_Id { get; set; }

        // Batting Stats
        public double B_G { get; set; }
        public double B_PA { get; set; }
        public double B_AB { get; set; }
        public double B_R { get; set; }
        public double B_H { get; set; }
        public double B_TB { get; set; }
        public double B_2B { get; set; }
        public double B_3B { get; set; }
        public double B_HR { get; set; }
        public double B_RBI { get; set; }
        public double B_BB { get; set; }
        public double B_IBB { get; set; }
        public double B_SO { get; set; }
        public double B_GDP { get; set; }
        public double B_HP { get; set; }
        public double B_SH { get; set; }
        public double B_SF { get; set; }
        public double B_SB { get; set; }
        public double B_CS { get; set; }
        public double B_XI { get; set; }
        public double B_PK { get; set; }
        public double B_FO { get; set; }
        public double B_GO { get; set; }

        // Pitching Stats
        public double P_G { get; set; }
        public double P_GS { get; set; }
        public double P_GO { get; set; }
        public double P_AO { get; set; }
        public double P_R { get; set; }
        public double P_2B { get; set; }
        public double P_3B { get; set; }
        public double P_HR { get; set; }
        public double P_SO { get; set; }
        public double P_BB { get; set; }
        public double P_IBB { get; set; }
        public double P_H { get; set; }
        public double P_HP { get; set; }
        public double P_AB { get; set; }
        public double P_CS { get; set; }
        public double P_SB { get; set; }
        public double P_PITCH { get; set; }
        public double P_OUT { get; set; }
        public double P_W { get; set; }
        public double P_L { get; set; }
        public double P_SV { get; set; }
        public double P_SVO { get; set; }
        public double P_HOLD { get; set; }
        public double P_BLSV { get; set; }
        public double P_ER { get; set; }
        public double P_TBF { get; set; }
        public double P_OUTS { get; set; }
        public double P_CG { get; set; }
        public double P_SHO { get; set; }
        public double P_PITCHES { get; set; }
        public double P_BALLS { get; set; }
        public double P_STRIKES { get; set; }
        public double P_HBP { get; set; }
        public double P_BK { get; set; }
        public double P_WP { get; set; }
        public double P_PK { get; set; }
        public double P_RBI { get; set; }
        public double P_GF { get; set; }
        public double P_IR { get; set; }
        public double P_IRS { get; set; }
        public double P_CI { get; set; }
        public double P_SH { get; set; }
        public double P_SF { get; set; }
        public double P_PB { get; set; }

        // Fielding Stats
        public double F_P_GS { get; set; }
        public double F_P_TC { get; set; }
        public double F_P_PO { get; set; }
        public double F_P_A { get; set; }
        public double F_P_E { get; set; }

        public double F_C_GS { get; set; }
        public double F_C_TC { get; set; }
        public double F_C_PO { get; set; }
        public double F_C_A { get; set; }
        public double F_C_E { get; set; }

        public double F_1B_GS { get; set; }
        public double F_1B_TC { get; set; }
        public double F_1B_PO { get; set; }
        public double F_1B_A { get; set; }
        public double F_1B_E { get; set; }

        public double F_2B_GS { get; set; }
        public double F_2B_TC { get; set; }
        public double F_2B_PO { get; set; }
        public double F_2B_A { get; set; }
        public double F_2B_E { get; set; }

        public double F_3B_GS { get; set; }
        public double F_3B_TC { get; set; }
        public double F_3B_PO { get; set; }
        public double F_3B_A { get; set; }
        public double F_3B_E { get; set; }

        public double F_SS_GS { get; set; }
        public double F_SS_TC { get; set; }
        public double F_SS_PO { get; set; }
        public double F_SS_A { get; set; }
        public double F_SS_E { get; set; }

        public double F_LF_GS { get; set; }
        public double F_LF_TC { get; set; }
        public double F_LF_PO { get; set; }
        public double F_LF_A { get; set; }
        public double F_LF_E { get; set; }

        public double F_CF_GS { get; set; }
        public double F_CF_TC { get; set; }
        public double F_CF_PO { get; set; }
        public double F_CF_A { get; set; }
        public double F_CF_E { get; set; }

        public double F_RF_GS { get; set; }
        public double F_RF_TC { get; set; }
        public double F_RF_PO { get; set; }
        public double F_RF_A { get; set; }
        public double F_RF_E { get; set; }

        // Target Variable
        public int Home_Win { get; set; }
    }
}
