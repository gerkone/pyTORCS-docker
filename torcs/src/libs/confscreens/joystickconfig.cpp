/***************************************************************************

    file                 : joystickconfig.cpp
    created              : Wed Mar 21 21:46:11 CET 2001
    copyright            : (C) 2001-2015 by Eric Espie, Bernhard Wymann
    email                : eric.espie@torcs.org
    version              : $Id: joystickconfig.cpp,v 1.5.2.7 2015/04/18 11:04:55 berniw Exp $

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <tgfclient.h>
#include <track.h>
#include <robot.h>
#include "driverconfig.h"
#include <playerpref.h>
#include <portability.h>
#include <plib/js.h>

#include "controlconfig.h"
#include "joystickconfig.h"

static void	*scrHandle2 = NULL;

static tCmdInfo *Cmd;
static int maxCmd;
static void *parmHandle;
static const char* driverSection;

static jsJoystick *js[NUM_JOY] = {NULL};

static float ax[_JS_MAX_AXES * NUM_JOY] = {0};
static int rawb[NUM_JOY] = {0};

#define NB_STEPS 6

#define OFFSET_CMD 5

static const char *Instructions[] = {
    "Center the joystick then press a button",
    "Steer left then press a button",
    "Steer right then press a button",
    "Apply full throttle then press a button",
    "Apply full brake then press a button",
    "Apply full clutch then press a button",
    "Calibration terminated",
    "Calibration failed"
};

static int CalState;
static int InstId;

static const char *LabName[] = { "Steer", "Throttle", "Brake", "Clutch" };
static int  LabAxisId[4];
static int  LabMinId[4];
static int  LabMaxId[4];


static void onBack(void *prevMenu)
{
    GfuiScreenActivate(prevMenu);
}


static float axCenter[_JS_MAX_AXES * NUM_JOY];

static void advanceStep (void)
{
	do {
		CalState++;
	} while ((Cmd[CalState + OFFSET_CMD].ref.type != GFCTRL_TYPE_JOY_AXIS) && (CalState < NB_STEPS));
}


static void JoyCalAutomaton(void)
{
	static int axis;
	const int BUFSIZE = 1024;
	char buf[BUFSIZE];
	
    switch (CalState) {
		case 0:
			memcpy(axCenter, ax, sizeof(axCenter));
			advanceStep();
			break;
		case 1:
			axis = Cmd[CalState + OFFSET_CMD].ref.index;
			Cmd[CalState + OFFSET_CMD].min = ax[axis];
			Cmd[CalState + OFFSET_CMD].max = axCenter[axis];
			Cmd[CalState + OFFSET_CMD].pow = 1.0;
			snprintf(buf, BUFSIZE, "%.2g", ax[axis]);
			GfuiLabelSetText(scrHandle2, LabMinId[0], buf);
			advanceStep();
			break;
		case 2:
			axis = Cmd[CalState + OFFSET_CMD].ref.index;
			Cmd[CalState + OFFSET_CMD].min = axCenter[axis];
			Cmd[CalState + OFFSET_CMD].max = ax[axis];
			Cmd[CalState + OFFSET_CMD].pow = 1.0;
			snprintf(buf, BUFSIZE, "%.2g", ax[axis]);
			GfuiLabelSetText(scrHandle2, LabMaxId[0], buf);
			advanceStep();
			break;
		case 3:
		case 4:
		case 5:
			axis = Cmd[CalState + OFFSET_CMD].ref.index;
			Cmd[CalState + OFFSET_CMD].min = axCenter[axis];
			Cmd[CalState + OFFSET_CMD].max = ax[axis];
			Cmd[CalState + OFFSET_CMD].pow = 1.0;
			snprintf(buf, BUFSIZE, "%.2g", axCenter[axis]);
			GfuiLabelSetText(scrHandle2, LabMinId[CalState - 2], buf);
			snprintf(buf, BUFSIZE, "%.2g", ax[axis]);
			GfuiLabelSetText(scrHandle2, LabMaxId[CalState - 2], buf);
			advanceStep();
			break;
    }
    GfuiLabelSetText(scrHandle2, InstId, Instructions[CalState]);
}


static void Idle2(void)
{
	int mask;
	int b, i;
	int index;

	for (index = 0; index < NUM_JOY; index++) {
		if (js[index]) {
			js[index]->read(&b, &ax[index * _JS_MAX_AXES]);
			
			/* Joystick buttons */
			for (i = 0, mask = 1; i < 32; i++, mask *= 2) {
				if (((b & mask) != 0) && ((rawb[index] & mask) == 0)) {
					const char *str = GfctrlGetNameByRef(GFCTRL_TYPE_JOY_BUT, i + 32 * index);
					if (!GfctrlIsEventBlacklisted(parmHandle, driverSection, str)) {
						/* Button fired */
						JoyCalAutomaton();
						if (CalState >= NB_STEPS) {
							glutIdleFunc(GfuiIdle);
						}
						glutPostRedisplay();
						rawb[index] = b;
						return;
					}
				}
			}
			rawb[index] = b;
		}
	}
}


static void onActivate(void * /* dummy */)
{
	int i;
	int index;
	int step;

	CalState = 0;
	GfuiLabelSetText(scrHandle2, InstId, Instructions[CalState]);
	glutIdleFunc(Idle2);
	glutPostRedisplay();
	for (index = 0; index < NUM_JOY; index++) {
		if (js[index]) {
			js[index]->read(&rawb[index], &ax[index * _JS_MAX_AXES]); /* initial value */
		}
	}

	for (i = 0; i < 4; i++) {
		if (i > 0) {
			step = i + 2;
		} else {
			step = i + 1;
		}
		if (Cmd[step + OFFSET_CMD].ref.type == GFCTRL_TYPE_JOY_AXIS) {
			GfuiLabelSetText(scrHandle2, LabAxisId[i], GfctrlGetNameByRef(GFCTRL_TYPE_JOY_AXIS, Cmd[step + OFFSET_CMD].ref.index));
		} else {
			GfuiLabelSetText(scrHandle2, LabAxisId[i], "---");
		}
		GfuiLabelSetText(scrHandle2, LabMinId[i], "");
		GfuiLabelSetText(scrHandle2, LabMaxId[i], "");
	}
}


void *JoyCalMenuInit(void *prevMenu, tCmdInfo *cmd, int maxcmd, void *parmhandle, const char* driversection)
{
	int x, y, dy, i, index;

	Cmd = cmd;
	maxCmd = maxcmd;
	parmHandle = parmhandle;
	driverSection = driversection;

	if (scrHandle2) {
		return scrHandle2;
	}

	scrHandle2 = GfuiScreenCreateEx(NULL, NULL, onActivate, NULL, NULL, 1);
	GfuiTitleCreate(scrHandle2, "Joystick Calibration", 0);
	GfuiMenuDefaultKeysAdd(scrHandle2);

	GfuiScreenAddBgImg(scrHandle2, "data/img/splash-joycal.png");

	x = 128;
	y = 300;
	dy = 50;

	for (i = 0; i < 4; i++) {
		GfuiLabelCreate(scrHandle2, LabName[i], GFUI_FONT_LARGE, x, y, GFUI_ALIGN_HC_VC, 0);
		LabAxisId[i] = GfuiLabelCreate(scrHandle2, "                ", GFUI_FONT_MEDIUM, 2 * x, y, GFUI_ALIGN_HC_VC, 0);
		LabMinId[i] = GfuiLabelCreate(scrHandle2,  "                ", GFUI_FONT_MEDIUM, 3 * x, y, GFUI_ALIGN_HC_VC, 0);
		LabMaxId[i] = GfuiLabelCreate(scrHandle2,  "                ", GFUI_FONT_MEDIUM, 4 * x, y, GFUI_ALIGN_HC_VC, 0);
		y -= dy;
	}

	for (index = 0; index < NUM_JOY; index++) {
		if (js[index] == NULL) {
			js[index] = new jsJoystick(index);
		}

		if (js[index]->notWorking()) {
			/* don't configure the joystick */
			js[index] = NULL;
		}
	}

	InstId = GfuiLabelCreate(scrHandle2, Instructions[0], GFUI_FONT_MEDIUM, 320, 80, GFUI_ALIGN_HC_VB, 60);

	GfuiButtonCreate(scrHandle2, "Back", GFUI_FONT_LARGE, 160, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				prevMenu, onBack, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);

	GfuiButtonCreate(scrHandle2, "Reset", GFUI_FONT_LARGE, 480, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, onActivate, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);

	return scrHandle2;
}