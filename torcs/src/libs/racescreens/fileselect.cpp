/***************************************************************************

    file        : fileselect.cpp
    created     : Sun Feb 16 13:09:23 CET 2003
    copyright   : (C) 2003-2014 by Eric Espie, Bernhard Wymann                        
    email       : eric.espie@torcs.org   
    version     : $Id: fileselect.cpp,v 1.2.2.5 2014/05/20 12:20:04 berniw Exp $                                  

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/** @file   
    Files selection screen.
    @author	Bernhard Wymann, Eric Espie
    @version	$Id: fileselect.cpp,v 1.2.2.5 2014/05/20 12:20:04 berniw Exp $
*/


#include <stdlib.h>
#include <stdio.h>
#include <sys/stat.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <tgfclient.h>
#include <racescreens.h>

static void *scrHandle = NULL;
static int fileScrollList;
static tRmFileSelect *rmFs;
static tFList *FileList = NULL;
static tFList *FileSelected;


static void rmActivate(void * /* dummy */ )
{
}


static void rmClickOnFile(void * /*dummy*/)
{
    GfuiScrollListGetSelectedElement(scrHandle, fileScrollList, (void**)&FileSelected);
}


static void rmSelect(void * /* dummy */ )
{
	if (FileList) {
		rmFs->select(FileSelected->name);
		GfDirFreeList(FileList, NULL, true, false);
		FileList = NULL;
	} else {
		rmFs->select(NULL);
	}
}

static void rmDeactivate(void * /* dummy */ )
{
	if (FileList) {
		GfDirFreeList(FileList, NULL, true, false);
		FileList = NULL;
	}
	GfuiScreenActivate(rmFs->prevScreen);
}


/** @brief File selection
 * 
 *  The files listed are the ones contained in the directory given by the path in tRmFileSelect.path
 *  @ingroup racemantools
 *  @param[in,out] vs Pointer on tRmFileSelect structure (cast to void)
 */
void RmFileSelect(void *vs)
{
	tFList *FileCur;

	rmFs = (tRmFileSelect*)vs;

	if (scrHandle) {
		GfuiScreenRelease(scrHandle);
	}
	scrHandle = GfuiScreenCreateEx((float*)NULL, NULL, rmActivate, NULL, (tfuiCallback)NULL, 1);
	GfuiScreenAddBgImg(scrHandle, "data/img/splash-filesel.png");
	GfuiTitleCreate(scrHandle, rmFs->title, 0);

	/* Scroll List containing the File list */
	fileScrollList = GfuiScrollListCreate(scrHandle, GFUI_FONT_MEDIUM_C, 120, 80, GFUI_ALIGN_HC_VB,
						400, 310, GFUI_SB_RIGHT, NULL, rmClickOnFile);

	FileList = GfDirGetList(rmFs->path);
	if (FileList == NULL) {
		GfuiScreenActivate(rmFs->prevScreen);
		return;
	}
	
	FileSelected = FileList;
	FileCur = FileList;
	do {
		FileCur = FileCur->next;
		GfuiScrollListInsertElement(scrHandle, fileScrollList, FileCur->name, 1000, (void*)FileCur);
	} while (FileCur != FileList);

	/* Bottom buttons */
	GfuiButtonCreate(scrHandle, "Select", GFUI_FONT_LARGE, 210, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmSelect, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);

	GfuiButtonCreate(scrHandle, "Cancel", GFUI_FONT_LARGE, 430, 40, 150, GFUI_ALIGN_HC_VB, GFUI_MOUSE_UP,
				NULL, rmDeactivate, NULL, (tfuiCallback)NULL, (tfuiCallback)NULL);

	/* Default menu keyboard actions */
	GfuiMenuDefaultKeysAdd(scrHandle);
	GfuiScreenActivate(scrHandle);
}
