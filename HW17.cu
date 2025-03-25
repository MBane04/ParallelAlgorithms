// Name: Mason Bane
// Two body problem
// nvcc HW17.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some not so crude that code moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.

 //user can now exit the simulation by hitting the 'q' key or the 'esc' key or the x in the window
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/freeglut.h> //if this doesn't work sudo apt-get install freeglut3-dev
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0

#define NUMBER_OF_SPHERES 10

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
bool killSim = false;

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
void setup();
void cleanUp();
void keyboard(unsigned char key, int x, int y);
int main(int, char**);

//lets use a struct for each sphere
struct SphereStruct
{
	float px, py, pz, vx, vy, vz, mass, fx, fy, fz;
};

SphereStruct *Spheres;

void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	//int yeahBuddy; Sorry but noBuddy needs you anymore :( not my fault bools are better
	float dx, dy, dz, seperation;

	//initialize the spheres
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		//random positions and velocities, mass is 1.0
		if( i == 0) //first sphere
		{
			Spheres[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Spheres[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Spheres[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		}
		else
		{
			//set the new sphere to a random position
			Spheres[i].px = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Spheres[i].py = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
			Spheres[i].pz = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;

			//check if the new sphere is too close to any other sphere
			for(int j = 0; j < i; j++)
			{
				bool tooClose = false;
				while(tooClose)
				{

					tooClose = false;
					dx = Spheres[j].px - Spheres[i].px;
					dy = Spheres[j].py - Spheres[i].py;
					dz = Spheres[j].pz - Spheres[i].pz;
					seperation = sqrt(dx*dx + dy*dy + dz*dz);
					if(seperation < DIAMETER)
					{
						tooClose = true;
						break;
					}
				}
				dx = Spheres[j].px - Spheres[i].px;
				dy = Spheres[j].py - Spheres[i].py;
				dz = Spheres[j].pz - Spheres[i].pz;
				seperation = sqrt(dx*dx + dy*dy + dz*dz);
				if(seperation < DIAMETER)
				{
					tooClose = true;
					break;
				}
			}
		}
        
		Spheres[i].vx = MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Spheres[i].vy = MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		Spheres[i].vz = MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
		
		Spheres[i].mass = 1.0;
	}

	
	// px1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	// py1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	// pz1 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	// yeahBuddy = 0;
	// while(yeahBuddy == 0)
	// {
	// 	px2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	// 	py2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
	// 	pz2 = (LENGTH_OF_BOX - DIAMETER)*rand()/RAND_MAX - (LENGTH_OF_BOX - DIAMETER)/2.0;
		
	// 	dx = px2 - px2;
	// 	dy = py2 - py1;
	// 	dz = pz2 - pz1;
	// 	seperation = sqrt(dx*dx + dy*dy + dz*dz);
	// 	yeahBuddy = 1;
	// 	if(seperation < DIAMETER) yeahBuddy = 0;
	// }
	
	// vx1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	// vy1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	// vz1 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
	// vx2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	// vy2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	// vz2 = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	
	// mass1 = 1.0;
	// mass2 = 1.0;
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}

void draw_picture()
{


	float radius = DIAMETER/2.0;
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    
    Drawwirebox();
    
    // Draw all spheres, alternating colors
    for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
        if(i % 2 == 0) 
		{
            glColor3d(1.0, 0.5, 1.0); // Pink color
        } 
		else 
		{
            glColor3d(0.0, 0.5, 0.0); // Green color
        }
        
        glPushMatrix();
        glTranslatef(Spheres[i].px, Spheres[i].py, Spheres[i].pz);
        glutSolidSphere(radius, 20, 20);
        glPopMatrix();
    }
    
    glutSwapBuffers();
	
	// glColor3d(1.0,0.5,1.0);
	// glPushMatrix();
	// glTranslatef(px1, py1, pz1);
	// glutSolidSphere(radius,20,20);
	// glPopMatrix();
	
	// glColor3d(0.0,0.5,0.0);
	// glPushMatrix();
	// glTranslatef(px2, py2, pz2);
	// glutSolidSphere(radius,20,20);
	// glPopMatrix();
	
	// glutSwapBuffers();
}

void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		if(Spheres[i].px > halfBoxLength)  //x pos
		{
			Spheres[i].px = 2.0*halfBoxLength - Spheres[i].px;
			Spheres[i].vx = - Spheres[i].vx;
		}
		else if(Spheres[i].px < -halfBoxLength) //x neg
		{
			Spheres[i].px = -2.0*halfBoxLength - Spheres[i].px;
			Spheres[i].vx = - Spheres[i].vx;
		}
		
		if(Spheres[i].py > halfBoxLength) //y pos
		{
			Spheres[i].py = 2.0*halfBoxLength - Spheres[i].py;
			Spheres[i].vy = - Spheres[i].vy;
		}
		else if(Spheres[i].py < -halfBoxLength) //y neg
		{
			Spheres[i].py = -2.0*halfBoxLength - Spheres[i].py;
			Spheres[i].vy = - Spheres[i].vy;
		}
				
		if(Spheres[i].pz > halfBoxLength) //z pos
		{
			Spheres[i].pz = 2.0*halfBoxLength - Spheres[i].pz;
			Spheres[i].vz = - Spheres[i].vz;
		}
		else if(Spheres[i].pz < -halfBoxLength) //z neg
		{
			Spheres[i].pz = -2.0*halfBoxLength - Spheres[i].pz;
			Spheres[i].vz = - Spheres[i].vz;
		}		
	}
	// if(px1 > halfBoxLength)
	// {
	// 	px1 = 2.0*halfBoxLength - px1;
	// 	vx1 = - vx1;
	// }
	// else if(px1 < -halfBoxLength)
	// {
	// 	px1 = -2.0*halfBoxLength - px1;
	// 	vx1 = - vx1;
	// }
	
	// if(py1 > halfBoxLength)
	// {
	// 	py1 = 2.0*halfBoxLength - py1;
	// 	vy1 = - vy1;
	// }
	// else if(py1 < -halfBoxLength)
	// {
	// 	py1 = -2.0*halfBoxLength - py1;
	// 	vy1 = - vy1;
	// }
			
	// if(pz1 > halfBoxLength)
	// {
	// 	pz1 = 2.0*halfBoxLength - pz1;
	// 	vz1 = - vz1;
	// }
	// else if(pz1 < -halfBoxLength)
	// {
	// 	pz1 = -2.0*halfBoxLength - pz1;
	// 	vz1 = - vz1;
	// }
	
	// if(px2 > halfBoxLength)
	// {
	// 	px2 = 2.0*halfBoxLength - px2;
	// 	vx2 = - vx2;
	// }
	// else if(px2 < -halfBoxLength)
	// {
	// 	px2 = -2.0*halfBoxLength - px2;
	// 	vx2 = - vx2;
	// }
	
	// if(py2 > halfBoxLength)
	// {
	// 	py2 = 2.0*halfBoxLength - py2;
	// 	vy2 = - vy2;
	// }
	// else if(py2 < -halfBoxLength)
	// {
	// 	py2 = -2.0*halfBoxLength - py2;
	// 	vy2 = - vy2;
	// }
			
	// if(pz2 > halfBoxLength)
	// {
	// 	pz2 = 2.0*halfBoxLength - pz2;
	// 	vz2 = - vz2;
	// }
	// else if(pz2 < -halfBoxLength)
	// {
	// 	pz2 = -2.0*halfBoxLength - pz2;
	// 	vz2 = - vz2;
	// }
}

void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;

	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		for(int j = i+1; j < NUMBER_OF_SPHERES; j++)
		{
			dx = Spheres[j].px - Spheres[i].px;
			dy = Spheres[j].py - Spheres[i].py;
			dz = Spheres[j].pz - Spheres[i].pz;
					
			r2 = dx*dx + dy*dy + dz*dz;
			r = sqrt(r2);
			
			forceMag =  Spheres[i].mass*Spheres[j].mass*GRAVITY/r2;

			if (r < DIAMETER)
			{
				dvx = Spheres[j].vx - Spheres[i].vx;
				dvy = Spheres[j].vy - Spheres[i].vy;
				dvz = Spheres[j].vz - Spheres[i].vz;
				inout = dx*dvx + dy*dvy + dz*dvz;
				if(inout <= 0.0)
				{
					forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
				else
				{
					forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
				}
			}

			Spheres[i].fx = forceMag*dx/r;
			Spheres[i].fy = forceMag*dy/r;
			Spheres[i].fz = forceMag*dz/r;
			Spheres[j].fx = -forceMag*dx/r;
			Spheres[j].fy = -forceMag*dy/r;
			Spheres[j].fz = -forceMag*dz/r;
		}
	}
	
	// dx = px2 - px1;
	// dy = py2 - py1;
	// dz = pz2 - pz1;
				
	// r2 = dx*dx + dy*dy + dz*dz;
	// r = sqrt(r2);

	// forceMag =  mass1*mass2*GRAVITY/r2;
			
	// if (r < DIAMETER)
	// {
	// 	dvx = vx2 - vx1;
	// 	dvy = vy2 - vy1;
	// 	dvz = vz2 - vz1;
	// 	inout = dx*dvx + dy*dvy + dz*dvz;
	// 	if(inout <= 0.0)
	// 	{
	// 		forceMag +=  SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
	// 	}
	// 	else
	// 	{
	// 		forceMag +=  PUSH_BACK_REDUCTION*SPHERE_PUSH_BACK_STRENGTH*(r - DIAMETER);
	// 	}
	// }

	// fx1 = forceMag*dx/r;
	// fy1 = forceMag*dy/r;
	// fz1 = forceMag*dz/r;
	// fx2 = -forceMag*dx/r;
	// fy2 = -forceMag*dy/r;
	// fz2 = -forceMag*dz/r;
}

void move_bodies(float time)
{
	for(int i = 0; i < NUMBER_OF_SPHERES; i++)
	{
		//i dont think we need a 2nd loop here, just go through the array of spheres 1 by 1. If the simulation acts weird I know where to look.
		if(time == 0)
		{
			Spheres[i].vx += 0.5*DT*(Spheres[i].fx - DAMP*Spheres[i].vx)/Spheres[i].mass;
			Spheres[i].vy += 0.5*DT*(Spheres[i].fy - DAMP*Spheres[i].vy)/Spheres[i].mass;
			Spheres[i].vz += 0.5*DT*(Spheres[i].fz - DAMP*Spheres[i].vz)/Spheres[i].mass;
		}
		else
		{
			Spheres[i].vx += DT*(Spheres[i].fx - DAMP*Spheres[i].vx)/Spheres[i].mass;
			Spheres[i].vy += DT*(Spheres[i].fy - DAMP*Spheres[i].vy)/Spheres[i].mass;
			Spheres[i].vz += DT*(Spheres[i].fz - DAMP*Spheres[i].vz)/Spheres[i].mass;
		}

		Spheres[i].px += DT*Spheres[i].vx;
		Spheres[i].py += DT*Spheres[i].vy;
		Spheres[i].pz += DT*Spheres[i].vz;
	}
	keep_in_box();

	// if(time == 0.0)
	// {
	// 	vx1 += 0.5*DT*(fx1 - DAMP*vx1)/mass1; // .5 * dt * (f - damp * v) / m (just to make sure i write the calculation correctly)
	// 	vy1 += 0.5*DT*(fy1 - DAMP*vy1)/mass1;
	// 	vz1 += 0.5*DT*(fz1 - DAMP*vz1)/mass1;
		
	// 	vx2 += 0.5*DT*(fx2 - DAMP*vx2)/mass2;
	// 	vy2 += 0.5*DT*(fy2 - DAMP*vy2)/mass2;
	// 	vz2 += 0.5*DT*(fz2 - DAMP*vz2)/mass2;
	// }
	// else
	// {
	// 	vx1 += DT*(fx1 - DAMP*vx1)/mass1;
	// 	vy1 += DT*(fy1 - DAMP*vy1)/mass1;
	// 	vz1 += DT*(fz1 - DAMP*vz1)/mass1;
		
	// 	vx2 += DT*(fx2 - DAMP*vx2)/mass2;
	// 	vy2 += DT*(fy2 - DAMP*vy2)/mass2;
	// 	vz2 += DT*(fz2 - DAMP*vz2)/mass2;
	// }

	// px1 += DT*vx1;
	// py1 += DT*vy1;
	// pz1 += DT*vz1;
	
	// px2 += DT*vx2;
	// py2 += DT*vy2;
	// pz2 += DT*vz2;
	

}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME && !killSim)
	{
		glutMainLoopEvent();// keep processing events
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;

	}

	cleanUp();
	exit(0);
	printf("\n DONE \n");
	//while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

void setup()
{
	Spheres = (SphereStruct*)malloc(NUMBER_OF_SPHERES * sizeof(SphereStruct));
	printf("\n\npress q, Q, or esc to quit\n");
}

void cleanUp()
{
	free(Spheres);
	printf("\n Memory freed, ur welcome. bye.\n ");
}

void keyPressed(unsigned char key, int x, int y)
{
    switch (key) 
	{
        case 27:    // Escape key
        case 'q':   // q key
        case 'Q':   // Q key
            printf("\n Exiting simulation...\n");
            killSim = true;  // Set the kill flag
            break;
        default:
            break;
    }
}

int main(int argc, char** argv)
{
	setup();
	set_initail_conditions();
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyPressed);
	glutMainLoop();
	atexit(cleanUp);
	return 0;
}


