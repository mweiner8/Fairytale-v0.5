import os
import time
import logging
import gevent
from datetime import datetime
from config import Config
from gevent import spawn, joinall
from image_processor import analyze_image, add_text_to_image
from story_loader import load_template_story, load_template_images
from ai_generator import generate_page_with_ai
from pdf_generator import create_simple_pdf
import db_utils

logger = logging.getLogger(__name__)


def generate_page_image(
        page_task,
        session_id,
        image_path,
        character_description,
        story_type,
        template_images,
        pages_dir,
        openai_client,
        socketio,
        progress_tracker
):
    """
    Generate a single page with retries and live progress updates.
    Returns a dict with all relevant info.
    """
    page_num = page_task['page_number']
    page_text = page_task['text']
    start_time = time.time()
    last_error = None

    try:
        gevent.sleep(0)

        # Select template
        template_img = template_images[0][1] if page_num == 0 else template_images[page_num][1]

        # Retry logic
        max_retries = 2
        complete_page = None
        for attempt in range(max_retries):
            try:
                complete_page = generate_page_with_ai(
                    template_img,
                    image_path,
                    page_text,
                    character_description,
                    story_type,
                    openai_client,
                    Config.UPLOAD_FOLDER
                )
                gevent.sleep(0)
                break
            except Exception as e:
                last_error = e
                time.sleep((attempt + 1) * 2)
                gevent.sleep(0)

        if complete_page is None:
            # Fallback if AI generation fails
            complete_page = add_text_to_image(template_img, page_text)
            last_error = f"Used fallback after {max_retries} attempts"

        # Save image
        page_name = 'cover' if page_num == 0 else f'page_{page_num}'
        page_path = os.path.join(pages_dir, f"{page_name}.png")
        complete_page.save(page_path, "PNG")

        elapsed = time.time() - start_time

        # Update progress tracker - use page_name as key to match original working code
        progress_tracker[session_id]['pages'][page_name] = {
            'page_number': page_num,
            'image_url': f'/page_image/{session_id}/{page_name}',
            'status': 'complete',
            'elapsed_time': elapsed
        }

        logger.info(f"✅ Page {page_num} completed in {elapsed:.2f}s")
        logger.info(
            f"   Stored as '{page_name}' in progress tracker. Total pages: {len(progress_tracker[session_id]['pages'])}")

        # Emit page-level completion immediately
        try:
            socketio.emit('page_complete', {
                'session_id': session_id,
                'page_number': page_num,
                'page_name': page_name,
                'image_url': f'/page_image/{session_id}/{page_name}',
                'status': 'complete'
            }, room=session_id)

            logger.info(f"📤 Emitted page_complete for page {page_num} to room {session_id}")
        except Exception as emit_error:
            logger.error(f"❌ Failed to emit page_complete for page {page_num}: {emit_error}")

        return {
            'page_num': page_num,
            'success': True,
            'result_image': complete_page,
            'error_msg': None,
            'elapsed_time': elapsed
        }

    except Exception as e:
        logger.error(f"Page {page_num} failed: {e}")
        return {
            'page_num': page_num,
            'success': False,
            'result_image': None,
            'error_msg': str(e),
            'elapsed_time': time.time() - start_time
        }


def generate_book_async(
        session_id,
        image_path,
        story_type,
        gender,
        child_name,
        user_id,
        progress_tracker,
        socketio,
        app,
        openai_client
):
    """
    Async book generation using gevent greenlets with batches and live updates.
    """
    with app.app_context():
        try:
            # Initialize progress - use page_name format to match original
            progress_tracker[session_id] = {
                'progress': 0,
                'status': 'Starting...',
                'error': None,
                'created_at': datetime.now().isoformat(),
                'completed': False,
                'pdf_path': None,
                'pages': {},  # Start empty, fill as pages complete
                'total_pages': 13
            }

            logger.info(f"Started book generation for session {session_id}")

            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': 0,
                'status': 'Starting...'
            }, room=session_id)

            # Step 1: Analyze child's image
            progress_tracker[session_id].update({
                'progress': 5,
                'status': "Analyzing child's photo..."
            })
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': 5,
                'status': "Analyzing child's photo..."
            }, room=session_id)

            character_description = analyze_image(image_path, openai_client)
            logger.info(f"Character analysis complete for session {session_id}")

            # Step 2: Load story
            progress_tracker[session_id].update({'progress': 10, 'status': 'Loading story template...'})
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': 10,
                'status': 'Loading story template...'
            }, room=session_id)

            story_data = load_template_story(story_type, child_name)
            template_images = load_template_images(story_type)
            logger.info(f"Story template loaded for session {session_id}")

            # Step 3: Prepare pages
            images_for_pdf = [None] * 13
            pages_dir = os.path.join(Config.OUTPUT_FOLDER, 'pages', session_id)
            os.makedirs(pages_dir, exist_ok=True)

            all_page_tasks = [
                {'page_number': 0, 'text': f"{story_data.get('title', '')}\n{story_data.get('subtitle', '')}", 'index': -1}
            ]
            for idx, page_data in enumerate(story_data.get('pages', [])):
                all_page_tasks.append({'page_number': idx + 1, 'text': page_data.get('text', ''), 'index': idx})

            total_pages = len(all_page_tasks)
            batch_size = 5
            completed_count = 0

            logger.info(f"Processing {total_pages} pages in batches of {batch_size}")

            # Step 4: Process in batches
            for batch_start in range(0, total_pages, batch_size):
                batch_tasks = all_page_tasks[batch_start:batch_start + batch_size]
                greenlets = []

                logger.info(f"Starting batch {batch_start // batch_size + 1} with {len(batch_tasks)} pages")

                for task in batch_tasks:
                    g = spawn(generate_page_image,
                              task,
                              session_id,
                              image_path,
                              character_description,
                              story_type,
                              template_images,
                              pages_dir,
                              openai_client,
                              socketio,
                              progress_tracker)
                    greenlets.append(g)

                joinall(greenlets)

                for g in greenlets:
                    result = g.value
                    page_num_result = result['page_num']
                    success = result['success']
                    result_image = result['result_image']
                    error_msg = result['error_msg']
                    elapsed = result['elapsed_time']

                    if success and result_image:
                        images_for_pdf[page_num_result] = result_image
                        completed_count += 1

                        # Emit overall progress
                        current_progress = 20 + int((completed_count / total_pages) * 70)
                        progress_tracker[session_id]['progress'] = current_progress
                        progress_tracker[session_id]['status'] = f'Completed {completed_count}/{total_pages} pages...'

                        socketio.emit('progress_update', {
                            'session_id': session_id,
                            'progress': current_progress,
                            'status': f'Completed {completed_count}/{total_pages} pages...'
                        }, room=session_id)

                        logger.info(f"Progress update: {completed_count}/{total_pages} pages complete ({current_progress}%)")
                    else:
                        logger.error(f"❌ Page {page_num_result} failed: {error_msg}")

            # Step 5: Filter successful pages
            images_for_pdf = [img for img in images_for_pdf if img is not None]
            if not images_for_pdf:
                raise ValueError("No pages were successfully generated")

            logger.info(f"All {len(images_for_pdf)} pages generated successfully")

            # Step 6: Create PDF
            progress_tracker[session_id]['progress'] = 95
            progress_tracker[session_id]['status'] = 'Creating PDF...'
            socketio.emit('progress_update', {
                'session_id': session_id,
                'progress': 95,
                'status': 'Creating PDF...'
            }, room=session_id)

            output_path = os.path.join(Config.OUTPUT_FOLDER, f'{session_id}.pdf')
            create_simple_pdf(images_for_pdf, output_path)
            logger.info(f"PDF created at {output_path}")

            # Step 7: Finalize progress
            progress_tracker[session_id].update({
                'progress': 100,
                'status': 'Complete!',
                'pdf_path': output_path,
                'completed': True,
                'completed_at': datetime.now().isoformat()
            })

            db_utils.update_book_status(session_id=session_id, status='completed', pdf_path=output_path)
            db_utils.create_log(level='INFO', message='Book generation completed successfully', session_id=session_id)

            socketio.emit('generation_complete', {
                'session_id': session_id,
                'pdf_path': output_path
            }, room=session_id)

            logger.info(f"Book generation complete for session {session_id}")

        except Exception as error:
            logger.error(f"Error in book generation for session {session_id}: {error}", exc_info=True)
            progress_tracker[session_id] = {
                'progress': 0,
                'status': f'Error: {str(error)}',
                'error': str(error),
                'completed': False,
                'pages': {}
            }
            db_utils.update_book_status(session_id=session_id, status='failed', error_message=str(error))
            db_utils.create_log(level='ERROR', message=f'Book generation failed: {str(error)}', session_id=session_id)

            socketio.emit('generation_error', {
                'session_id': session_id,
                'error': str(error)
            }, room=session_id)