
#if !defined _CLASSES_H_
#define _CLASSES_H_

#include <math.h>
#include <complex>


using namespace std;

inline
double vector_dot(double x[], double y[])
{
    return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
}
//
inline
void vector_cross(double x[], double y[], double z[])
{
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
}

void GL_1D( int n, double x[], double w[] );

class Quadrature
{

public:
	int N;
	double* w;    // Pointer to array of integration  weights
    double* z; // Pointer to array of integration  points

	Quadrature()
	{
		N = 0;
		*w = NULL;
		*z = NULL;
	}

	Quadrature(int Np_1D)
	{
		N = Np_1D;
		w = new double [Np_1D];
		z = new double [Np_1D];
		GL_1D(N, z, w);
	}

	~Quadrature()
	{
		delete[] w;
		delete[] z;
		N = 0;
	}

};

class Geometry
{
//private:
	

public:
    double rp1[3],rp2[3],rp3[3],rp4[3];
    double rq1[3],rq2[3],rq3[3],rq4[3];
	double rp_c[3];
	double rq_c[3];
	double delta;
    double k0;
	double nq[3], np[3];
	int lp, lq;
	int kerneltype;

	//default
    Geometry()
    {
        *rp_c = NULL; 
		*rp_c = NULL; 
        *np = NULL;   
        *nq = NULL;   
		delta = 0.0;  
        k0 = 0.0;    
        kerneltype = 0;
        lp = 0;
       lq = 0;
    };
    
   
	// ST case
	void ST(const double r1[], const double r2[], const double r3[], const double r4[])
	{
		
        k0 = 0.0;
		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r1[i];
				rp2[i] = r2[i];
				rp3[i] = r3[i];
				rp4[i] = r4[i];
        
				rq1[i] = r1[i];
				rq2[i] = r2[i];
				rq3[i] = r3[i];
				rq4[i] = r4[i];
			}
	}

	// EA case
	void EA(double r1[],  double r2[], double r3[],  double r4[], double r5[], double r6[])
	{

		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r4[i];
				rp2[i] = r3[i];
				rp3[i] = r6[i];
				rp4[i] = r5[i];
        
				rq1[i] = r3[i];
				rq2[i] = r4[i];
				rq3[i] = r1[i];
				rq4[i] = r2[i];
			}	
	}

	// VA case
	void VA(const double r1[], const double r2[], const double r3[], const double r4[],const double r5[],const double r6[], const double r7[])
	{
	
		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r3[i];
				rp2[i] = r5[i];
				rp3[i] = r6[i];
				rp4[i] = r7[i];
        
				rq1[i] = r3[i];
				rq2[i] = r4[i];
				rq3[i] = r1[i];
				rq4[i] = r2[i];
			}	
	}
    
	void set_centers(double r1[], double r2[])
	{
		for (int i = 0; i < 3; i++)
			{
				rq_c[i] = r1[i];
				rp_c[i] = r2[i];
		    }
	}
    void set_normales(double n1[],double n2[])
    {
        for (int i = 0; i < 3; i++)
		{
			nq[i] = n1[i];
			np[i] = n2[i];
		}
    }
	void set_delta(double del)
	{
		delta = del;
	}
     
    void set_wavenumber(double k)
    {
        k0 = k;
    }
	void set_kerneltype(int ktype)
    {
        kerneltype = ktype;
    }
	void set_lq(int l)
	{
		lq = l;
	}
	void set_lp(int l)
	{
		lp = l;
	}
	double Jacobians(double u_p, double v_p, double u_q, double v_q )
	{
		double Ep[3],Gp[3],Jp;
	   double Eq[3],Gq[3],Jq;
           
		for (int i = 0; i < 3; i++)
		{
			Ep[i] = -rp1[i] + rp2[i] + rp3[i] - rp4[i] + v_p*(rp1[i] - rp2[i] + rp3[i] - rp4[i]);
			Gp[i] = -rp1[i] - rp2[i] + rp3[i] + rp4[i] + u_p*(rp1[i] - rp2[i] + rp3[i] - rp4[i]);

			Eq[i] = -rq1[i] + rq2[i] + rq3[i] - rq4[i] + v_q*(rq1[i] - rq2[i] + rq3[i] - rq4[i]);
			Gq[i] = -rq1[i] - rq2[i] + rq3[i] + rq4[i] + u_q*(rq1[i] - rq2[i] + rq3[i] - rq4[i]);
		}
	    
        
		Jp = sqrt(vector_dot(Ep,Ep)*vector_dot(Gp,Gp) - vector_dot(Ep,Gp)*vector_dot(Ep,Gp))/double(16.0);
		Jq = sqrt(vector_dot(Eq,Eq)*vector_dot(Gq,Gq) - vector_dot(Eq,Gq)*vector_dot(Eq,Gq))/double(16.0);
        
		return Jp*Jq;
	}

	void position_vectors(double u_p, double v_p, double u_q, double v_q, double rp[], double rq[])
	{
		
		for (int i = 0; i < 3; i++)
		{
			rp[i] = (rp1[i] + rp2[i] + rp3[i] +rp4[i] + u_p*(-rp1[i] + rp2[i] + rp3[i] - rp4[i]) + v_p*(-rp1[i] - rp2[i] + rp3[i] +rp4[i]) + u_p*v_p*(rp1[i] - rp2[i] + rp3[i] - rp4[i]))/double(4.0);
			rq[i] = (rq1[i] + rq2[i] + rq3[i] +rq4[i] + u_q*(-rq1[i] + rq2[i] + rq3[i] - rq4[i]) + v_q*(-rq1[i] - rq2[i] + rq3[i] +rq4[i]) + u_q*v_q*(rq1[i] - rq2[i] + rq3[i] - rq4[i]))/double(4.0);
		}

	}
	void normales(double np[], double nq[])
	{
		double lp1[3];
		double lp2[3];
		double lq1[3];
		double lq2[3];

		double n1[3],n2[3];

		for (int i = 0; i < 3; i++)
		{
			lp1[i] = rp2[i] - rp1[i];
			lp2[i] = rp3[i] - rp1[i];
			lq1[i] = rq2[i] - rq1[i];
			lq2[i] = rq3[i] - rq1[i];
		}
		vector_cross(lp1,lp2,n1);
		vector_cross(lq2,lq2,n2);

		for(int i = 0; i < 3; i++)
		{
			np[i] = n1[i]/vector_dot(n1,n1);
			nq[i] = n2[i]/vector_dot(n2,n2);
		}

	}
};

class Geometry_triangle
{
private:
	double rp1[3],rp2[3],rp3[3];
    double rq1[3],rq2[3],rq3[3];

public:

	double rp_c[3];
	double rq_c[3];
	double delta;
    double k0;
	double nq[3],np[3];
	int kerneltype;
	int lq, lp;

	//default
    Geometry_triangle()
    {
        *rp_c = NULL; 
		*rp_c = NULL; 
        *np = NULL;   
        *nq = NULL;   
		delta = 0.0;  
        k0 = 0.0;    
        kerneltype = 0;
        lp = 0;
       lq = 0;
    };
	
	// ST case
	void ST(const double r1[], const double r2[], const double r3[])
	{
		
        k0 = 0.0;
		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r1[i];
				rp2[i] = r2[i];
				rp3[i] = r3[i];
        
				rq1[i] = r1[i];
				rq2[i] = r2[i];
				rq3[i] = r3[i];
			}
	}

	// EA case
	void EA(const double r1[], const double r2[], const double r3[], const double r4[])
	{
		//*rp_n = NULL;
		//*rp_n = NULL;
		//delta = 0.0;
        k0 = 0.0;
		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r1[i];
				rp2[i] = r2[i];
				rp3[i] = r3[i];
			
				rq1[i] = r2[i];
				rq2[i] = r1[i];
				rq3[i] = r4[i];
			}	
	};

	// VA case
	void VA(const double r1[], const double r2[], const double r3[], const double r4[],const double r5[])
	{
		//*rp_n = NULL;
		//*rp_n = NULL;
		//delta = 0.0;
        k0 = 0.0;
		for (int i = 0; i < 3; i++)
			{
				rp1[i] = r1[i];
				rp2[i] = r2[i];
				rp3[i] = r3[i];
        
				rq1[i] = r1[i];
				rq2[i] = r4[i];
				rq3[i] = r5[i];
			}	
	};
    void set_lq(int l)
	{
		lq = l;
	}
	void set_lp(int l)
	{
		lp = l;
	}
    void set_wavenumber(double k)
    {
        k0 = k;
    }
	void set_centers(double r1[], double r2[])
	{
		for (int i = 0; i < 3; i++)
			{
				rq_c[i] = r1[i];
				rp_c[i] = r2[i];
		    }
	}
    void set_normales(double n1[],double n2[])
    {
        for (int i = 0; i < 3; i++)
		{
			nq[i] = n1[i];
			np[i] = n2[i];
		}
    }
	void set_delta(double del)
	{
		delta = del;
	}
     void set_kerneltype(int ktype)
    {
        kerneltype = ktype;
    }
   
	/*
	double Jacobians(double u_p, double v_p, double u_q, double v_q)
	{
		double Ep[3],Gp[3],Jp;
	    double Eq[3],Gq[3],Jq;

		for (int i = 0; i < 3; i++)
		{
			Ep[i] = -rp1[i] + rp2[i] + rp3[i] - rp4[i] + v_p*(rp1[i] - rp2[i] + rp3[i] - rp4[i]);
			Gp[i] = -rp1[i] - rp2[i] + rp3[i] + rp4[i] + u_p*(rp1[i] - rp2[i] + rp3[i] - rp4[i]);

			Eq[i] = -rq1[i] + rq2[i] + rq3[i] - rq4[i] + v_q*(rq1[i] - rq2[i] + rq3[i] - rq4[i]);
			Gq[i] = -rq1[i] - rq2[i] + rq3[i] + rq4[i] + u_q*(rq1[i] - rq2[i] + rq3[i] - rq4[i]);
		}
	
		Jp = sqrt(vector_dot(Ep,Ep)*vector_dot(Gp,Gp) - vector_dot(Ep,Gp)*vector_dot(Ep,Gp))/double(16.0);
		Jq = sqrt(vector_dot(Eq,Eq)*vector_dot(Gq,Gq) - vector_dot(Eq,Gq)*vector_dot(Eq,Gq))/double(16.0);

		return Jp*Jq;
	}
	*/
	void position_vectors(double xi_1p, double xi_2p, double xi_3p, double xi_1q, double xi_2q, double xi_3q , double rp[], double rq[])
	{
		
		for (int i = 0; i < 3; i++)
		{
			 rp[i]     = xi_1p * rp1[i] + xi_2p * rp2[i] + xi_3p * rp3[i];
             rq[i]     = xi_1q * rq1[i] + xi_2q * rq2[i] + xi_3q * rq3[i];
		}

	}
	void normales(double np[], double nq[])
	{
		double lp1[3];
		double lp2[3];
		double lq1[3];
		double lq2[3];

		double n1[3],n2[3];

		for (int i = 0; i < 3; i++)
		{
			lp1[i] = rp2[i] - rp1[i];
			lp2[i] = rp3[i] - rp1[i];
			lq1[i] = rq2[i] - rq1[i];
			lq2[i] = rq3[i] - rq1[i];
		}
		vector_cross(lp1,lp2,n1);
		vector_cross(lq2,lq2,n2);

		for(int i = 0; i < 3; i++)
		{
			np[i] = n1[i]/vector_dot(n1,n1);
			nq[i] = n2[i]/vector_dot(n2,n2);
		}

	}
};
#endif