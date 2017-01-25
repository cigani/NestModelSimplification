#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CaDynamics_E2_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTa_t_reg(void);
extern void _NaTs2_t_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPANMDA_EMS_reg(void);
extern void _ProbGABAAB_EMS_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/CaDynamics_E2.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/Ca_HVA.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/Ca_LVAst.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/Ih.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/Im.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/K_Pst.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/K_Tst.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/NaTa_t.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/NaTs2_t.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/Nap_Et2.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/ProbAMPANMDA_EMS.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/ProbGABAAB_EMS.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/SK_E2.mod");
    fprintf(stderr," ../../Documents/Models/L5_TTPC1_cADpyr232_1/mechanisms/SKv3_1.mod");
    fprintf(stderr, "\n");
  }
  _CaDynamics_E2_reg();
  _Ca_HVA_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _Im_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTa_t_reg();
  _NaTs2_t_reg();
  _Nap_Et2_reg();
  _ProbAMPANMDA_EMS_reg();
  _ProbGABAAB_EMS_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
}
